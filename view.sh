# mpv output/*depth.png --merge-files=yes --container-fps-override=24 --no-correct-pts --msg-level=demux=error

mpv "mf://output/*depth.png" --mf-fps=24 --external-file="mf://output/*frame.png" --mf-fps=24 --lavfi-complex="[vid2][vid1]vstack[vo]" --no-correct-pts --msg-level=demux=error --autofit=50%