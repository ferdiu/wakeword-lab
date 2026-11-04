"""Optional Edge TTS corpus generator. Requires edge-tts and ffmpeg."""
import asyncio
from pathlib import Path
import csv
import edge_tts

async def _save(text,voice,rate,destination):
    await edge_tts.Communicate(text,voice,rate=rate).save(str(destination))

async def synthesize(project,phrase):
    p=Path(project); out=p/'data/raw/positive'; out.mkdir(parents=True,exist_ok=True)
    voices=[('en-IN',3),('en-US',1),('en-GB',1),('en-AU',1)]
    listing=await edge_tts.list_voices(); selected=[]
    for locale,weight in voices:
        matches=[v for v in listing if v['Locale'].startswith(locale)]
        for v in matches[:max(1,weight*2)]: selected.append(v)
    rows=[]
    for i,v in enumerate(selected):
        name=f"tts_{v['ShortName'].replace('-','_')}"
        path=out/f'{name}.mp3'
        await _save(phrase,v['ShortName'],'+0%',path)
        rows.append((path.name,name))
    # Metadata is written; convert MP3 to mono 16k WAV with ffmpeg before prepare.
    with (p/'data/raw/tts_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['path','speaker_id']); w.writerows(rows)
    return len(rows)

def main():
    import argparse
    a=argparse.ArgumentParser(); a.add_argument('project'); a.add_argument('--phrase',required=True); x=a.parse_args()
    print(asyncio.run(synthesize(x.project,x.phrase)))
if __name__=='__main__': main()
