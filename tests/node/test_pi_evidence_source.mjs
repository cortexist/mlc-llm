import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { refreshStaleEvidence } from '../../deploy/xavier/pi-evidence-source.mjs';

test('stale hit relocates only an unchanged unique passage in a bounded local file',()=>{
 const dir=mkdtempSync(join(tmpdir(),'evidence-source-'));
 const passage='Available memory was 15032 MiB with three resident workers.';
 try {
  const path=join(dir,'notes.md');
  writeFileSync(path,'New heading\n\n'+passage+'\nThis is a historical snapshot.\n');
  const source={path,text:passage,stale:true,start_line:99};
  const result=refreshStaleEvidence(source,dir);
  assert.equal(result.stale,false);
  assert.equal(result.start_line,1);
  assert.match(result.text,/historical snapshot/);
  writeFileSync(path,'New measurement differs from the cached passage.');
  assert.equal(refreshStaleEvidence(source,dir),null);
  writeFileSync(path,passage+'\n'+passage);
  assert.equal(refreshStaleEvidence(source,dir),null);
  writeFileSync(path,'x'.repeat(65537)+passage);
  assert.equal(refreshStaleEvidence(source,dir),null);
  writeFileSync(path,passage);
  const link=join(dir,'linked.md');symlinkSync(path,link);
  assert.equal(refreshStaleEvidence({...source,path:link},dir),null);
  assert.equal(refreshStaleEvidence({...source,path:'/etc/passwd'},dir),null);
 } finally {rmSync(dir,{recursive:true,force:true});}
});
