"""Create the immutable raw-text BGE-M3 cache for Experiment 11."""
from __future__ import annotations
import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw/paired_states_v2/paired_states.json"; OUT=ROOT/"runs/2026-10-01_exp11_bgem3/embeddings"; NPZ=OUT/"bge_m3.npz"; INDEX=OUT/"bge_m3.index.json"; META=OUT/"bge_m3.metadata.json"; URL="http://127.0.0.1:8001/v1/embeddings"
def sha(b):return hashlib.sha256(b).hexdigest()
def embed(texts):
 data=json.dumps({"model":"BAAI/bge-m3","input":texts,"max_length":8192}).encode();req=urllib.request.Request(URL,data=data,headers={"Content-Type":"application/json"});return np.asarray([x["embedding"] for x in json.loads(urllib.request.urlopen(req,timeout=120).read())["data"]],dtype=np.float32)
def main():
 if NPZ.exists() or INDEX.exists() or META.exists():raise FileExistsError("Refusing overwrite BGE cache")
 OUT.mkdir(parents=True,exist_ok=True);rows=json.loads(RAW.read_text(encoding="utf-8"));rows=sorted(rows,key=lambda x:x["uid"]);texts=[];index=[]
 for kind in ("state_old","state_new"):
  for r in rows:
   texts.append(r[kind])
   index.append({"kind":kind,"id":r["uid"],"state_id" if kind=="state_old" else "query_id":r["uid"],"text_sha256":sha(r[kind].encode())})
 for r in rows:
  for j,text in enumerate(r["possible_invalidation_conditions"][:4],1):texts.append(text);index.append({"kind":"pic","id":f"{r['uid']}:PIC_{j}","state_id":r["uid"],"pic_id":f"{r['uid']}:PIC_{j}","text_sha256":sha(text.encode())})
 vectors=np.concatenate([embed(texts[i:i+4]) for i in range(0,len(texts),4)])
 norms=np.linalg.norm(vectors,axis=1)
 if vectors.shape!=(1200,1024) or len(index)!=1200 or not np.isfinite(vectors).all() or not np.allclose(norms,1,atol=1e-3):
  raise ValueError(f"BGE cache validation failed: shape={vectors.shape}, entries={len(index)}, finite={np.isfinite(vectors).all()}, norm_min={norms.min()}, norm_max={norms.max()}")
 np.savez_compressed(NPZ,embeddings=vectors);INDEX.write_text(json.dumps({"schema_version":"experiment11-bge-m3-index-v1","entries":index},indent=2)+"\n",encoding="utf-8")
 META.write_text(json.dumps({"created_at_utc":datetime.now(timezone.utc).isoformat(),"model":"BAAI/bge-m3","checkpoint":"models/bge-m3","embedding_dimension":1024,"count":{"state_old":200,"state_new":200,"pic":800},"tokenizer":"AutoTokenizer(local_files_only=true)","pooling":"CLS token","normalization":"L2","cosine":"dot product of L2-normalized vectors","input_policy":"raw text only; no instruction prefix","cache_finite":True,"norm_min":float(np.linalg.norm(vectors,axis=1).min()),"norm_max":float(np.linalg.norm(vectors,axis=1).max()),"dataset_sha256":sha(RAW.read_bytes())},indent=2)+"\n",encoding="utf-8")
 print(NPZ)
if __name__=="__main__":main()
