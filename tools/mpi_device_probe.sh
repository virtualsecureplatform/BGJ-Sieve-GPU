#!/bin/bash
set -u
hostname
id
echo "memlock_kib=$(ulimit -l)"
ls -ln /dev/infiniband
grep -E '^(Cap|NoNewPrivs|Seccomp)' /proc/self/status || true
grep -E 'infiniband|/dev ' /proc/self/mountinfo || true
python3 -c 'import glob,os
for path in glob.glob("/dev/infiniband/*"):
    try:
        fd=os.open(path,os.O_RDWR|os.O_CLOEXEC); os.close(fd)
        print(path,"open OK")
    except OSError as e:
        print(path,"open",e.errno,str(e))'
