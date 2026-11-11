import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EvidenceRouter, evidenceAction, coverageAction } from '../../deploy/xavier/pi-evidence-router.mjs';
const answer = (choice, probabilities) => ({choice,probabilities});
const ordinary = answer('general',{workspace:.02,inventory:.03,general:.94,action:.01});

test('semantic probabilities route ambiguous local facts, general advice and actions', () => {
 assert.equal(evidenceAction(ordinary),'agent');
 assert.equal(evidenceAction(answer('general',{workspace:.2,inventory:.02,general:.77,action:.01})),'workspace');
 assert.equal(evidenceAction(answer('action',{workspace:.3,inventory:.02,general:.08,action:.6})),'agent');
 assert.equal(evidenceAction(answer('inventory',{workspace:.1,inventory:.8,general:.09,action:.01})),'inventory');
 assert.throws(()=>evidenceAction(answer('general',{workspace:NaN})),/Invalid/);
});

test('bounded cache avoids duplicate inference and separates environments', async () => {
 const dir=await mkdtemp(join(tmpdir(),'d1-router-'));
 try {
  await writeFile(join(dir,'key'),'test-key');
  await writeFile(join(dir,'config'),JSON.stringify({baseUrl:'https://server.example/v1',keyFile:join(dir,'key')}));
  let calls=0;
  const router=new EvidenceRouter({configPath:join(dir,'config'),maxEntries:1,fetchImpl:async (url,opts)=>{
   calls++;assert.equal(url,'https://server.example/v1/decisions');assert.equal(opts.redirect,'error');
   assert.equal(opts.headers.Authorization,'Bearer test-key');
   return new Response(JSON.stringify({answers:{route:ordinary},usage:{input_tokens:180}}),{headers:{'X-Xavier-Decision-Bucket-Tokens':'256'}});
  }});
  assert.equal((await router.route('What is an embedding?','A')).cached,false);
  assert.equal((await router.route('What is an embedding?','A')).cached,true);
  assert.equal(calls,1);
  await router.route('What is an embedding?','B');
  await router.route('What is an embedding?','A');
  assert.equal(calls,3);assert.equal(router.cache.size,1);
  assert.equal((await router.route('/quit','A')).reason,'outside_router_scope');
  assert.equal(calls,3);
 } finally {await rm(dir,{recursive:true,force:true});}
});

test('unavailable or malformed classifier leaves tools with the agent',async()=>{
 const router=new EvidenceRouter({configPath:'/path/that/does/not/exist'});
 assert.equal((await router.route('Find our benchmark','A')).action,null);
 assert.equal((await router.route('x'.repeat(1601),'A')).reason,'outside_router_scope');
});

test('coverage decisions are cached against actual inventory and fail to one search',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'d1-coverage-'));
 try {
  await writeFile(join(dir,'key'),'test-key');
  const configPath=join(dir,'config');
  await writeFile(configPath,JSON.stringify({baseUrl:'https://server.example/v1',keyFile:join(dir,'key')}));
  let calls=0, malformed=false, expectedTokens=4096;
  const router=new EvidenceRouter({configPath,fetchImpl:async (_url,opts)=>{
   calls++;
   const body=JSON.parse(opts.body);
   assert.equal(body.questions.coverage.type,'choice');
   assert.deepEqual(body.state.current_deployment[0],{id:'example',task:'chat',context_length:expectedTokens});
   const coverage=malformed ? {choice:'inventory',probabilities:{inventory:NaN}} :
     {choice:'search',probabilities:{inventory:.03,search:.97}};
   return new Response(JSON.stringify({answers:{coverage}}));
  }});
  const models=[{id:'example',context_length:4096,private_field:'not sent'}];
  assert.equal((await router.inventoryCoverage('Recorded RAM?','local',models)).action,'workspace');
  assert.equal((await router.inventoryCoverage('Recorded RAM?','local',models)).cached,true);
  assert.equal(calls,1);
  expectedTokens=8192;
  assert.equal((await router.inventoryCoverage('Recorded RAM?','local',[{id:'example',context_length:8192}])).cached,false);
  assert.equal(calls,2);
  expectedTokens=4096;
  malformed=true;
  const bad=await router.inventoryCoverage('Other question','local',models);
  assert.equal(bad.action,'workspace');assert.equal(bad.reason,'router_unavailable');
  assert.equal(calls,3); // No internal retry or recursive classification.
  assert.equal(coverageAction({choice:'inventory',probabilities:{inventory:.8,search:.2}}),'inventory');
 } finally {await rm(dir,{recursive:true,force:true});}
});
