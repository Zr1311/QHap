#!/usr/bin/env python3
"""Download and verify selected HG002 BAM/BAI assets using ALIGNMENTS.json."""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request,time
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fetch(url,dest,expected=None):
    if dest.exists() and expected and sha(dest)==expected:return
    part=dest.with_name(dest.name+'.partial')
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'QHap-benchmark-downloader'})
            with urllib.request.urlopen(req,timeout=60) as r,part.open('wb') as w:shutil.copyfileobj(r,w,8*1024*1024)
            if expected and sha(part)!=expected:raise ValueError('Checksum mismatch: '+dest.name)
            part.replace(dest);return
        except (OSError,ValueError):
            if attempt==2:raise
            time.sleep(2)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',choices=['HiFi','ONT','CycloneSEQ'],required=True)
    p.add_argument('--region','--chromosome',dest='region',choices=['MHC','chr6','chr22'],required=True,
                   help='Select a verified manifest entry: MHC is a derived overlap subset; chr6/chr22 are complete supplied chromosome inputs')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'HG002_CHM13v2/ALIGNMENTS.json').read_text())
    records=[r for r in manifest['alignments'] if f'_{a.platform}_' in r['file'] and r.get('region',r.get('chromosome'))==a.region]
    if len(records)!=2:raise ValueError('Expected one BAM and its matching BAI')
    for record in records:
        target=a.output/record['file']
        if target.exists() and sha(target)==record['sha256']:
            print('Already verified:',target.name);continue
        pieces=[]
        for item in record['parts']:
            if isinstance(item,str):
                base=record['existing_release'].replace('/tag/','/download/')
                name=item;url=base+'/'+item;digest=None
            else:name=item['name'];url=item['url'];digest=item['sha256']
            local=a.output/name;fetch(url,local,digest);pieces.append(local)
        # Never truncate a downloaded one-part BAM while restoring it.
        tmp=target.with_name(target.name+'.restoring')
        with tmp.open('wb') as w:
            for piece in pieces:
                with piece.open('rb') as r:shutil.copyfileobj(r,w,8*1024*1024)
        if tmp.stat().st_size!=record['bytes'] or sha(tmp)!=record['sha256']:
            raise ValueError('Restored checksum mismatch: '+target.name)
        tmp.replace(target);print('Verified:',target.name)
        for piece in pieces:
            if piece!=target:piece.unlink()
if __name__=='__main__':main()
