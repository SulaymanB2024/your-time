"""Accounting and missing-data checks for the production browser helpers."""

import json
import subprocess

import pytest

from behavior_analysis import aggregate
from dashboard_explore import CONTEXT_JS
from dashboard_fixture import fixture


def evaluate(body, payload=None):
    script = CONTEXT_JS + "const fixtureData=JSON.parse(require('fs').readFileSync(0,'utf8'));\n" + body
    result = subprocess.run(['node', '-e', script], input=json.dumps(payload or fixture()),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_calendar_week_keeps_seven_dates_and_missing_days_across_dst():
    result = evaluate("""
const records=[{day:'2026-10-30'},{day:'2026-11-05'}];
console.log(JSON.stringify({ends:weekEndKeys(records),week:calendarWeek(records,'2026-11-05')}));
""")
    assert result['ends'] == [f'2026-10-{d}' for d in (30, 31)] + [f'2026-11-0{d}' for d in range(1, 6)]
    assert len(result['week']) == 7
    assert [d['day'] for d in result['week']] == result['ends']
    assert sum(not d.get('missing') for d in result['week']) == 2
    assert result['week'][2] == {'day': '2026-11-01', 'missing': True}


def test_exact_windows_replace_broad_context_without_adding_time():
    data = fixture()
    result = evaluate("""
const day=fixtureData.days.at(-1),rows=contextRows(day);
console.log(JSON.stringify({rows,total:rows.reduce((s,r)=>s+r.sampled_seconds,0),
 unknown:contextDaySeconds(day,{status:'unknown'}),period:contextRows(day,false)}));
""", data)
    assert result['total'] == pytest.approx(data['days'][-1]['mac_active_seconds'])
    assert result['unknown'] == 540
    assert any(r['label'] == 'Project notes' and r['status'] == 'observed_window' for r in result['rows'])
    assert not any(r['label'] == 'Other window context' for r in result['rows'])
    assert result['period'] == sorted(data['days'][-1]['workstreams'], key=lambda r: -r['sampled_seconds'])


def test_inconsistent_sessions_fall_back_and_compact_absence_stays_unknown():
    result = evaluate("""
const day=fixtureData.days.at(-1),window={label:'Project notes',status:'observed_window'};
const valid=contextDaySeconds(day,window),compact=fixtureData.history[0];
day.behavior.sessions[0].sampled_seconds+=600;
console.log(JSON.stringify({valid,fallback:contextRows(day),badSeconds:contextDaySeconds(day,window),
 compact:contextDaySeconds(compact,window),missing:contextDaySeconds({missing:true},window),
 original:mergeContexts(day.workstreams)}));
""")
    assert result['valid'] == 300
    assert result['fallback'] == result['original']
    assert result['badSeconds'] is None
    assert result['compact'] is None
    assert result['missing'] is None


def test_historical_week_totals_and_continuity_use_its_own_records():
    data = fixture()
    result = evaluate("""
const week=calendarWeek([...fixtureData.history,...fixtureData.days],'2026-09-20');
console.log(JSON.stringify({count:week.filter(d=>!d.missing).length,contexts:periodContexts(week),
 behavior:aggregateContinuity(week),latest:aggregateContinuity(fixtureData.days)}));
""", data)
    assert result['count'] == 1
    assert sum(r['sampled_seconds'] for r in result['contexts']) == data['history'][0]['mac_active_seconds']
    expected = aggregate(data['history'])
    for key in ('identified_seconds', 'context_switches', 'session_count',
                'longest_stretch_seconds', 'sustained_seconds',
                'context_switches_per_identified_hour', 'distribution'):
        assert result['behavior'][key] == expected[key]
    assert result['behavior']['identified_seconds'] < result['latest']['identified_seconds']


def test_exploration_intervals_exclude_idle_and_preserve_phone_union():
    data = fixture()
    result = evaluate("""
const rows=foregroundRecords(fixtureData.days.at(-1));
console.log(JSON.stringify({mac:rows.filter(r=>r.device==='Mac').reduce((s,r)=>s+r.sampled_seconds,0),
phone:rows.filter(r=>r.device==='iPhone').reduce((s,r)=>s+r.sampled_seconds,0),
 unknown:rows.filter(r=>r.status==='unknown').reduce((s,r)=>s+r.sampled_seconds,0),
 apps:rows.find(r=>r.label==='Project notes').apps,
 compact:foregroundRecords(fixtureData.history[0])}));
""", data)
    assert result['mac'] == data['days'][-1]['mac_foreground_seconds']
    assert result['phone'] == 840
    assert result['unknown'] == 540
    assert result['apps'] == ['Editor']
    assert result['compact'] == []


def test_period_inspection_joins_recent_detail_without_changing_saved_totals():
    result = evaluate("""
const current=fixtureData.days.at(-1),saved={day:current.day,mac_active_seconds:600,
workstreams:[{label:'Atlas',status:'specific_model',sampled_seconds:600}]};
const detail=contextDetailDays([saved,{day:'2026-10-05',missing:true}],fixtureData.days,false);
console.log(JSON.stringify({saved:contextDaySeconds(saved,{label:'Atlas',status:'specific_model'},false),
 detailCount:detail.length,intervalCount:foregroundRecords(detail[0]).length,
exact:contextDetailDays([saved],fixtureData.days,true)[0].mac_active_seconds}));
""")
    assert result['saved'] == 600
    assert result['detailCount'] == 1
    assert result['intervalCount'] == 22
    assert result['exact'] == 600
