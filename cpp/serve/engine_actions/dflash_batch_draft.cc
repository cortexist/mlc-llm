/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/dflash_batch_draft.cc
 * \brief DFlash batch draft action: single-pass block diffusion draft.
 *  Unlike EAGLE's iterative drafting, DFlash generates all block_size tokens
 *  in a single forward pass through the lightweight draft model.
 */

#include <tvm/runtime/device_api.h>

#include <cmath>
#include <iomanip>
#include <numeric>

#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"
#include "dflash_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs single-pass block draft proposal for DFlash.
 *  Model 0 = target (provides embed + lm_head), Model 1 = DFlash draft (stateless).
 */
class DFlashBatchDraftActionObj : public EngineActionObj {
 public:
  explicit DFlashBatchDraftActionObj(Array<Model> models, LogitProcessor logit_processor,
                                     Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                     DraftTokenWorkspaceManager draft_token_workspace_manager,
                                     EngineConfig engine_config,
                                     Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run when there are two models and running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt if needed — only check target model (model 0) pages.
    // DFlash draft model has no KV cache pages.
    std::vector<RequestStateEntry> running_rsentries = estate->GetRunningRequestStateEntries();
    while (!CanDecode(running_rsentries.size())) {
      if (estate->prefix_cache->TryFreeMemory()) continue;
      RequestStateEntry preempted = PreemptLastRunningRequestStateEntry(
          estate, models_, draft_token_workspace_manager_, trace_recorder_);
      if (preempted.same_as(running_rsentries.back())) {
        running_rsentries.pop_back();
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    int num_rsentries = running_rsentries.size();
    if (num_rsentries == 0) {
      return {};
    }

    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    Array<RequestModelState> mstates;
    request_ids.reserve(num_rsentries);
    request_internal_ids.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    mstates.reserve(num_rsentries);
    for (const RequestStateEntry& rsentry : running_rsentries) {
      request_ids.push_back(rsentry->request->id);
      request_internal_ids.push_back(rsentry->mstates[0]->internal_id);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
      // Use draft model's mstate (model 1) for draft token tracking,
      // consistent with verify which reads from mstates[draft_model_id_].
      mstates.push_back(rsentry->mstates[1]);
    }

    int block_size = estate->spec_draft_length;
    TVM_FFI_ICHECK_GT(block_size, 0) << "DFlash spec_draft_length (block_size) must be positive.";
    if (block_size == 1) return {};  // Verification consumes only the frontier.
    int num_draft = block_size - 1;

    // DFlash single-pass draft: process each request independently through the
    // stateless draft model.
    {
      for (int req_idx = 0; req_idx < num_rsentries; ++req_idx) {
        auto tdraft_start = std::chrono::high_resolution_clock::now();
        int64_t internal_id = request_internal_ids[req_idx];

        // Check that projected_target_hidden exists for this request.
        auto& pth_map = *model_workspaces_[0].dflash_projected_target_hidden;
        auto it = pth_map.find(internal_id);
        if (it == pth_map.end()) {
          continue;
        }
        const auto settings = models_[1]->GetDFlashSettings();
        Tensor projected_target_hidden = it->second.View();
        int context_offset = 0;
        if (settings.sliding_window > 0) {
          const int keep = settings.sliding_window - 1;
          context_offset = std::max<int64_t>(0, projected_target_hidden.shape()[1] - keep);
          if (context_offset > 0) {
            const int64_t width = projected_target_hidden.shape()[2];
            auto dtype = projected_target_hidden.dtype();
            projected_target_hidden =
                tvm::runtime::CreateView(projected_target_hidden, {1, keep, width}, dtype,
                                         context_offset * width * GetDataTypeBytes(dtype));
            projected_target_hidden = CompactView(projected_target_hidden, &draft_context_input_);
          }
        }
        if (DFlashDebugEnabled()) {
          LOG(WARNING) << "[DFlash-draft] projected_target_hidden shape=["
                       << projected_target_hidden.shape()[0] << ","
                       << projected_target_hidden.shape()[1] << ","
                       << projected_target_hidden.shape()[2]
                       << "] dtype=" << projected_target_hidden.dtype();
          {
            TensorStats pth_stats = ComputeTensorStats(projected_target_hidden);
            LOG(WARNING) << "[DFlash-draft] " << FormatTensorStats("projected_ctx", pth_stats);
          }

          LOG(WARNING) << "[DFlash-draft] req=" << req_idx << " internal_id=" << internal_id
                       << " ctx_len=" << projected_target_hidden.shape()[1]
                       << " block_size=" << block_size;
        }

        const int mask_token_id = settings.mask_token_id;

        // Build input tokens: first token is the committed token, rest are mask tokens.
        int first_token;
        if (!mstates[req_idx]->draft_output_tokens.empty()) {
          first_token = mstates[req_idx]->draft_output_tokens.back().GetTokenId();
        } else {
          TVM_FFI_ICHECK(!mstates[req_idx]->committed_tokens.empty());
          first_token = mstates[req_idx]->committed_tokens.back().GetTokenId();
        }
        std::vector<int> block_tokens(block_size, mask_token_id);
        block_tokens[0] = first_token;

        // 1. Embed via target model.
        RECORD_EVENT(trace_recorder_, request_ids, "start dflash draft embedding");
        ObjectRef noise_embedding =
            models_[0]->TokenEmbed({Shape{block_tokens.begin(), block_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish dflash draft embedding");

        // Reshape embedding from [block_size, hidden_size] to [1, block_size, hidden_size].
        Tensor embed_nd = noise_embedding.as_or_throw<Tensor>();
        int hidden_size = embed_nd.shape()[1];
        ObjectRef reshaped_embedding =
            tvm::runtime::CreateView(embed_nd, {1, block_size, hidden_size}, embed_nd.dtype());

        // 2. Determine context length from projected_target_hidden shape.
        int ctx_len = projected_target_hidden.shape()[1];
        int total_len = ctx_len + block_size;

        // 3. Construct RoPE cos/sin and attention mask.
        const DFlashDraftAuxTensors& aux_tensors =
            GetDraftAuxTensors(total_len, context_offset, projected_target_hidden.device());

        // 4. DFlash draft forward (single-pass through stateless draft model).
        RECORD_EVENT(trace_recorder_, request_ids, "start dflash draft forward");
        ObjectRef draft_hidden =
            models_[1]->DFlashDraftForward(reshaped_embedding, projected_target_hidden,
                                           aux_tensors.cos, aux_tensors.sin, aux_tensors.mask);
        RECORD_EVENT(trace_recorder_, request_ids, "finish dflash draft forward");

        // 5. Get logits via target model's LM head (shared).
        // Reshape draft_hidden from [1, block_size, h] to [block_size, h] for GetLogits.
        Tensor draft_hidden_nd = draft_hidden.as_or_throw<Tensor>();
        if (DFlashDebugEnabled()) {
          TensorStats dh_stats = ComputeTensorStats(draft_hidden_nd);
          LOG(WARNING) << "[DFlash-draft] " << FormatTensorStats("draft_hidden_out", dh_stats);
        }
        // Position zero is the known frontier. Avoid its expensive vocabulary
        // projection: only masked positions produce draft proposals.
        Tensor proposed_hidden = tvm::runtime::CreateView(
            draft_hidden_nd, {num_draft, hidden_size}, draft_hidden_nd.dtype(),
            hidden_size * GetDataTypeBytes(draft_hidden_nd.dtype()));
        // Compiled TIR requires byte_offset=0. Copy this small activation slice
        // into reusable storage rather than projecting the frontier vocabulary.
        if (!draft_head_input_.defined() || draft_head_input_.shape()[0] != num_draft) {
          draft_head_input_ = tvm::runtime::Empty({num_draft, hidden_size}, draft_hidden_nd.dtype(),
                                                  draft_hidden_nd.device());
        }
        tvm::runtime::CopyFromTo(proposed_hidden.GetDLTensorPtr(),
                                 draft_head_input_.GetDLTensorPtr());
        Tensor reshaped_hidden = draft_head_input_;
        Tensor logits;
        bool use_draft_logits = models_[1]->CanGetLogits();
        if (use_draft_logits) {
          logits = models_[1]->GetLogits(reshaped_hidden);
        } else {
          logits = models_[0]->GetLogits(reshaped_hidden);
        }
        // Handle possible 3D output [1, block_size, v].
        if (logits.ndim() == 3) {
          logits = tvm::runtime::CreateView(logits, {logits.shape()[1], logits.shape()[2]},
                                            logits.dtype());
        }
        TVM_FFI_ICHECK_EQ(logits.ndim(), 2);
        TVM_FFI_ICHECK_EQ(logits.shape()[0], num_draft);

        // 6. Apply logit processing for all block_size positions at once.
        // Replicate config for each position (all belong to same request).
        Array<String> rep_request_ids;
        Array<GenerationConfig> rep_gen_cfg;
        Array<RequestModelState> rep_mstates;
        rep_request_ids.reserve(num_draft);
        rep_gen_cfg.reserve(num_draft);
        rep_mstates.reserve(num_draft);
        for (int d = 0; d < num_draft; ++d) {
          rep_request_ids.push_back(request_ids[req_idx]);
          rep_gen_cfg.push_back(generation_cfg[req_idx]);
          rep_mstates.push_back(mstates[req_idx]);
        }
        // DFlash block diffusion generates all positions simultaneously — no sequential
        // draft token penalty needed. Pass nullptr for draft_mstates/draft_token_indices.
        logit_processor_->InplaceUpdateLogits(logits, rep_gen_cfg, rep_mstates, rep_request_ids);
        // 7. Sample from positions 1..block_size-1 (skip position 0 = committed token).
        bool deterministic_generation = IsDeterministicGenerationConfig(generation_cfg[req_idx]);
        Tensor probs_on_device{nullptr};
        std::vector<SampleResult> sample_results;
        if (deterministic_generation) {
          std::vector<int32_t> greedy_tokens = models_[0]->BatchArgmax(logits);
          sample_results.reserve(num_draft);
          for (int d = 0; d < num_draft; ++d) {
            sample_results.push_back(SampleResult{{greedy_tokens[d], 1.0f}, {}});
          }
          // Log draft model's top-5 per position
          if (DFlashDebugEnabled()) {
            Tensor logits_on_host = CopyTensorToCPU(logits);
            TensorStats logit_stats = ComputeTensorStats(logits_on_host);
            LOG(WARNING) << "[DFlash-draft] " << FormatTensorStats("draft_logits", logit_stats);
            for (int d = 0; d < std::min(num_draft, 8); ++d) {
              auto topk = TopKLogits(logits_on_host, d, 5);
              LOG(WARNING) << "[DFlash-draft] pos=" << (d + 1)
                           << " chosen=" << sample_results[d].GetTokenId()
                           << " top5=" << FormatTopK(topk);
            }
          }
        } else {
          probs_on_device =
              logit_processor_->ComputeProbsFromLogits(logits, rep_gen_cfg, rep_request_ids);
          std::vector<int> sample_indices;
          Array<String> sample_request_ids;
          Array<GenerationConfig> sample_gen_cfg;
          std::vector<RandomGenerator*> sample_rngs;
          sample_indices.reserve(num_draft);
          sample_request_ids.reserve(num_draft);
          sample_gen_cfg.reserve(num_draft);
          sample_rngs.reserve(num_draft);
          for (int d = 0; d < num_draft; ++d) {
            sample_indices.push_back(d);
            sample_request_ids.push_back(request_ids[req_idx]);
            sample_gen_cfg.push_back(generation_cfg[req_idx]);
            sample_rngs.push_back(rngs[req_idx]);
          }

          Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
              probs_on_device, sample_indices, sample_request_ids, sample_gen_cfg);
          sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
              renormalized_probs, sample_indices, sample_request_ids, sample_gen_cfg, sample_rngs);
          TVM_FFI_ICHECK_EQ(static_cast<int>(sample_results.size()), num_draft);
        }

        // 8. Store draft probs and add draft tokens.
        draft_token_slots_.clear();
        draft_token_workspace_manager_->AllocSlots(num_draft, &draft_token_slots_);
        if (!deterministic_generation) {
          models_[0]->ScatterDraftProbs(probs_on_device, draft_token_slots_,
                                        &model_workspaces_[0].draft_probs_storage);
        }

        // Add block_size-1 draft tokens as a linear chain.
        for (int d = 0; d < num_draft; ++d) {
          int64_t parent_idx = d - 1;  // -1 for first draft, 0 for second, etc.
          mstates[req_idx]->AddDraftToken(sample_results[d], draft_token_slots_[d], parent_idx);
        }

        auto tdraft_end = std::chrono::high_resolution_clock::now();
        estate->metrics.UpdateDraftTimeByBatchSize(
            1, static_cast<double>((tdraft_end - tdraft_start).count()) / 1e9);
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_decode_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  Tensor draft_head_input_{nullptr};
  Tensor draft_context_input_{nullptr};
  Tensor cos_input_{nullptr};
  Tensor sin_input_{nullptr};

  Tensor CompactView(const Tensor& view, Tensor* scratch) {
    if (view.GetDLTensorPtr()->byte_offset == 0) return view;
    if (!scratch->defined() || !std::equal(view.shape().begin(), view.shape().end(),
                                           scratch->shape().begin(), scratch->shape().end())) {
      *scratch = tvm::runtime::Empty(view.shape(), view.dtype(), view.device());
    }
    tvm::runtime::CopyFromTo(view.GetDLTensorPtr(), scratch->GetDLTensorPtr());
    return *scratch;
  }

  bool CanDecode(int num_rsentries) {
    // Only check target model (model 0) — DFlash draft model has no KV cache.
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  DFlashDraftAuxTensors CreateDraftAuxTensors(int total_len, int block_size, DLDevice device) {
    const auto settings = models_[1]->GetDFlashSettings();
    const int head_dim = settings.head_dim;
    const double rope_theta = settings.rope_theta;

    int half_dim = head_dim / 2;
    int total_elements = total_len * head_dim;

    std::vector<float> cos_data(total_elements);
    std::vector<float> sin_data(total_elements);

    for (int i = 0; i < half_dim; ++i) {
      double inv_freq = 1.0 / std::pow(rope_theta, 2.0 * i / head_dim);
      for (int p = 0; p < total_len; ++p) {
        double angle = p * inv_freq;
        float cos_val = static_cast<float>(std::cos(angle));
        float sin_val = static_cast<float>(std::sin(angle));
        // Duplicate: positions i and i+half_dim have the same cos/sin.
        cos_data[p * head_dim + i] = cos_val;
        cos_data[p * head_dim + i + half_dim] = cos_val;
        sin_data[p * head_dim + i] = sin_val;
        sin_data[p * head_dim + i + half_dim] = sin_val;
      }
    }

    Shape shape{1, total_len, 1, head_dim};
    Tensor cos = tvm::runtime::Empty(shape, (DLDataType{kDLFloat, 32, 1}), device);
    Tensor sin = tvm::runtime::Empty(shape, (DLDataType{kDLFloat, 32, 1}), device);
    tvm::runtime::CopyFromBytes(cos, cos_data.data(), cos_data.size() * sizeof(float));
    tvm::runtime::CopyFromBytes(sin, sin_data.data(), sin_data.size() * sizeof(float));

    Shape mask_shape({1, 1, block_size, total_len});
    Tensor mask = tvm::runtime::Empty(mask_shape, (DLDataType{kDLFloat, 32, 1}), device);
    size_t num_bytes = block_size * total_len * sizeof(float);
    std::vector<char> zeros(num_bytes, 0);
    tvm::runtime::CopyFromBytes(mask, zeros.data(), num_bytes);
    // Ensure async host-to-device copies complete before local buffers are freed.
    tvm::runtime::DeviceAPI::Get(device)->StreamSync(device, nullptr);
    return {cos, sin, mask};
  }

  DFlashDraftAuxTensors GetDraftAuxTensors(int total_len, int position_offset, DLDevice device) {
    // Exact-length caching retains O(context^2) storage as decoding advances.
    // Retain just one geometric allocation, and return contiguous prefix views.
    auto& storage = (*model_workspaces_[0].dflash_draft_aux_tensors)[0];
    int absolute_length = total_len + position_offset;
    if (!storage.cos.defined() || storage.cos.shape()[1] < absolute_length) {
      int64_t capacity = storage.cos.defined() ? storage.cos.shape()[1] : 256;
      while (capacity < absolute_length) capacity *= 2;
      storage = CreateDraftAuxTensors(capacity, engine_config_->spec_draft_length, device);
    }
    const auto dtype = storage.cos.dtype();
    int head_dim = storage.cos.shape()[3];
    int block_size = engine_config_->spec_draft_length;
    return {CompactView(tvm::runtime::CreateView(storage.cos, {1, total_len, 1, head_dim}, dtype,
                                                 position_offset * head_dim * sizeof(float)),
                        &cos_input_),
            CompactView(tvm::runtime::CreateView(storage.sin, {1, total_len, 1, head_dim}, dtype,
                                                 position_offset * head_dim * sizeof(float)),
                        &sin_input_),
            // The mask is all zero; its contiguous prefix can be reshaped safely.
            // Model-specific sliding-window masking is computed on device.
            tvm::runtime::CreateView(storage.mask, {1, 1, block_size, total_len}, dtype)};
  }

  Array<Model> models_;
  LogitProcessor logit_processor_;
  Sampler sampler_;
  std::vector<ModelWorkspace> model_workspaces_;
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  EngineConfig engine_config_;
  Optional<EventTraceRecorder> trace_recorder_;
  std::vector<int> draft_token_slots_;
};

EngineAction EngineAction::DFlashBatchDraft(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<DFlashBatchDraftActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
