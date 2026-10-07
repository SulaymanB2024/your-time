import json
import os
import subprocess
import sys

from tools.repo_layout import source_file


def synthetic_control(tmp_path):
    commands = tmp_path/'bin'
    commands.mkdir()
    home = tmp_path/'home'
    agents = home/'Library/LaunchAgents'
    agents.mkdir(parents=True)
    labels = ('com.yourtime.secure-screen-record', 'com.yourtime.secure-mac-activity',
              'com.yourtime.secure-window-reader')
    for label in labels:
        (agents/(label+'.plist')).write_text('synthetic installed agent')
    launchctl = commands/'launchctl'
    launchctl.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_STATE'])
d=json.loads(p.read_text())
action=sys.argv[1]; label=sys.argv[-1].split('/')[-1]
d['calls'].append([action,label])
if action=='print':
    if label not in d['jobs']: p.write_text(json.dumps(d)); sys.exit(1)
    print('state = '+d['jobs'][label])
elif action in ('kickstart','bootstrap'): d['jobs'][label]='running'
p.write_text(json.dumps(d))
''')
    launchctl.chmod(0o700)
    mock = tmp_path/'mock'
    mock.mkdir()
    (mock/'shutil.py').write_text('from types import SimpleNamespace\ndef disk_usage(_): return SimpleNamespace(free=20*1024**3)\n')
    state = tmp_path/'state.json'
    state.write_text(json.dumps({'jobs': {labels[0]:'not running', labels[1]:'running', labels[2]:'running'}, 'calls': []}))
    env = dict(os.environ, HOME=str(home), PATH=str(commands)+':/usr/bin:/bin',
               FAKE_STATE=str(state), PYTHONPATH=str(mock))
    return state, env, labels


def test_loaded_stopped_is_not_reported_as_running(tmp_path):
    _, env, labels = synthetic_control(tmp_path)
    script = source_file('capture_control.zsh')
    result = subprocess.run(['/bin/zsh', str(script), 'status'], env=env,
                            capture_output=True, text=True, timeout=5, check=True)
    assert labels[0]+' loaded_stopped' in result.stdout
    assert labels[1]+' running' in result.stdout


def test_resume_kickstarts_loaded_stopped_job_without_interrupting_running_jobs(tmp_path):
    state, env, labels = synthetic_control(tmp_path)
    script = source_file('capture_control.zsh')
    subprocess.run(['/bin/zsh', str(script), 'resume'], env=env,
                   capture_output=True, text=True, timeout=5, check=True)
    data = json.loads(state.read_text())
    assert data['jobs'][labels[0]] == 'running'
    assert [c for c in data['calls'] if c[0]=='kickstart'] == [['kickstart',labels[0]]]
