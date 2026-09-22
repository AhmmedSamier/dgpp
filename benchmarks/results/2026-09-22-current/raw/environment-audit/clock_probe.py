import json,subprocess,time
r={'epoch_ns':time.time_ns()}
for name,cmd in [('timesync',['timedatectl','show','-p','NTPSynchronized','-p','NTP']),('chrony',['chronyc','tracking'])]:
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=5);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except OSError as e:r[name]={'error':str(e)}
print(json.dumps(r))
