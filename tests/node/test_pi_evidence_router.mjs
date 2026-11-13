import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EvidenceRouter, evidenceAction, coverageAction } from '../../deploy/xavier/pi-evidence-router.mjs';
const reply = (content, usage) => new Response(JSON.stringify({choices:[{message:{role:'assistant',content}}],usage}));

test('option names route local facts and inventory; general advice and actions stay with the agent', () => {
 assert.equal(evidenceAction('general'),'agent');
 assert.equal(evidenceAction('action'),'agent');
 assert.equal(evidenceAction('workspace'),'workspace');
 assert.equal(evidenceAction('inventory'),'inventory');
 assert.throws(()=>evidenceAction('search'),/Invalid/);
 assert.throws(()=>evidenceAction('constructor'),/Invalid/);
});

test('bounded cache avoids duplicate inference and separates environments', async () => {
 const dir=await mkdtemp(join(tmpdir(),'d1-router-'));
 try {
  await writeFile(join(dir,'key'),'test-key');
  await writeFile(join(dir,'config'),JSON.stringify({baseUrl:'https://server.example/v1',keyFile:join(dir,'key')}));
  let calls=0;
  const router=new EvidenceRouter({configPath:join(dir,'config'),maxEntries:1,fetchImpl:async (url,opts)=>{
   calls++;assert.equal(url,'https://server.example/v1/chat/completions');assert.equal(opts.redirect,'error');
   assert.equal(opts.headers.Authorization,'Bearer test-key');
   assert.equal(opts.headers['X-Xavier-Scratch'],'1');
   const body=JSON.parse(opts.body);
   assert.deepEqual([body.model,body.temperature,body.max_tokens,body.messages.length],['nemotron',0,8,1]);
   assert.match(body.messages[0].content,/^Classify the user intent\..*\n\nOptions:\n- workspace: .*\n- action: .*\n\nEnvironment: [AB]\nRequest: What is an embedding\?\n\nReply with one word: the option name\.$/s);
   return reply('**General.**\n',{prompt_tokens:528});
  }});
  const first=await router.route('What is an embedding?','A');
  assert.deepEqual([first.action,first.choice,first.cached,first.input_tokens],['agent','general',false,528]);
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
 const dir=await mkdtemp(join(tmpdir(),'router-reply-'));
 try {
  await writeFile(join(dir,'key'),'test-key');
  await writeFile(join(dir,'config'),JSON.stringify({baseUrl:'https://server.example/v1',keyFile:join(dir,'key')}));
  let calls=0;
  const talkative=new EvidenceRouter({configPath:join(dir,'config'),fetchImpl:async()=>{calls++;return reply('I would search the workspace.');}});
  const bad=await talkative.route('Find our benchmark','A');
  assert.deepEqual([bad.action,bad.reason],[null,'router_unavailable']);
  await talkative.route('Find our benchmark','A');
  assert.equal(calls,2); // A reply that names no option is not cached.
 } finally {await rm(dir,{recursive:true,force:true});}
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
   const content=JSON.parse(opts.body).messages[0].content;
   assert.ok(content.includes(`\nEnvironment: local\nCurrent deployment: [{"id":"example","task":"chat","context_length":${expectedTokens}}]\nRequest: `));
   assert.ok(!content.includes('private_field'));
   return reply(malformed ? 'maybe' : 'search');
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
  assert.equal(coverageAction('inventory'),'inventory');
  assert.throws(()=>coverageAction('workspace'),/Invalid/);
 } finally {await rm(dir,{recursive:true,force:true});}
});
