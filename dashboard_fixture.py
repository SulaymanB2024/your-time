"""Synthetic-only design fixture. Never reads the user's activity database."""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from behavior_analysis import build, compact, aggregate
from local_dashboard import activity_score, time_of_day
from window_topic_tagging import window_id
from dashboard_timeline import build_timeline


def fixture() -> dict:
    now = datetime.now(timezone.utc)
    days = []
    for age in range(6, -1, -1):
        day = date(2026, 10, 4) - timedelta(days=age)
        at = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=14)
        rows = []
        tasks = [(2100, 'Editor', 'Atlas design', 'Atlas'),
                 (1800, 'Browser', 'Atlas research', 'Atlas'),
                 (300, 'Editor', 'Project notes', None),
                 (90, 'Browser', 'Reading list', None),
                 (1500, 'Editor', 'Orbit analysis', 'Orbit'),
                 (900, 'Browser', 'Orbit planning', 'Orbit'),
                 (60, 'Editor', 'Quick notes', None),
                 (2100-age*120, 'Editor', 'Atlas build', 'Atlas')]
        topics = []
        for index, (seconds, app, title, topic) in enumerate(tasks):
            if index in (2, 5):
                end = at + timedelta(seconds=300)
                rows.append(dict(start_utc=at.isoformat(),end_utc=end.isoformat(),
                                 state='idle',sampled_seconds=300,app=None,window=None))
                at = end
            end = at + timedelta(seconds=seconds)
            rows.append(dict(start_utc=at.isoformat(),end_utc=end.isoformat(),
                             state='active',sampled_seconds=seconds,app=app,window=title))
            if topic:
                topics.append(dict(id=window_id(app,title),topic=topic,status='model_inference'))
            at = end
        end = at + timedelta(seconds=540)
        rows.append(dict(start_utc=at.isoformat(),end_utc=end.isoformat(),state='unattributed',
                         sampled_seconds=540,app='Browser',window=None))
        phone = [dict(start_utc=(at+timedelta(minutes=i)).isoformat(),
                      end_utc=(at+timedelta(minutes=i+1)).isoformat(),app=f'Example app {i+1}') for i in range(14)]
        report = dict(day_local=day.isoformat(),mac={'segments':rows},iphone={'sessions':phone})
        behavior = build(rows,topics)
        totals = defaultdict(float)
        for row, task in zip([r for r in rows if r['state']=='active'],tasks):
            totals[(task[3] or 'Other window context', 'specific_model' if task[3] else 'broad_context')] += row['sampled_seconds']
        active = behavior['identified_seconds']
        days.append(dict(day=day.isoformat(),complete_day=age>0,data_quality={'status':'passed'},
                         mac_active_seconds=active,mac_foreground_seconds=active+540,
                         mac_unattributed_seconds=540,mac_app_only_seconds=540,mac_idle_seconds=600,
                         mac_locked_seconds=36000,mac_unobserved_seconds=12000,
                         iphone_focus_seconds=840,iphone_intervals=14,
                         mac_apps=[{'app':'Editor','name':'Editor','seconds':active*.6},
                                   {'app':'Browser','name':'Browser','seconds':active*.4+540}],
                         iphone_apps=[{'app':s['app'],'name':s['app'],'focus_seconds':60} for s in phone],
                         behavior=behavior,score=activity_score(report,{t['id']:t['topic'] for t in topics},now),
                         timeline=build_timeline(report,behavior,lambda app: app),
                         hourly=time_of_day(report),workstreams=[dict(label=label,status=status,sampled_seconds=value)
                             for (label,status),value in totals.items()],
                         screen_workstreams=[],user_task_labels=[],task_outcomes=[],browser_domains=[],
                         calendar={},work_artifacts={'local_commits':3+age,'changed_files':14+age,'repositories':[]},
                         blocks=[],screenshots=100,ocr_complete=100,ocr_pending=0,screen_context_status='complete'))
        days[-1]['review_candidates'] = [dict(id='fixture-gap',day=day.isoformat(),
            start_utc=at.isoformat(),end_utc=end.isoformat(),unknown_seconds=540,
            suggested_label='Possible reading',confirmed_label=None,outcome_marked=False)]
    history = [{**days[0], 'day':'2026-09-20','compact_only':True,'score':[], 'timeline':None,
                'hourly':{},'behavior':compact(days[0]['behavior']),'review_candidates':[],
                'iphone_apps':[{'app':x['app'],'name':x['name'],'seconds':x['focus_seconds']}
                               for x in days[0]['iphone_apps']]}]
    months=[]
    for key in ('2026-09','2026-10'):
        included=[d for d in history+days if d['day'].startswith(key)]
        streams=defaultdict(float)
        for d in included:
            for w in d['workstreams']:streams[(w['label'],w['status'])]+=w['sampled_seconds']
        months.append(dict(key=key,days_recorded=len(included),mac_active_seconds=sum(d['mac_active_seconds'] for d in included),
                           mac_foreground_seconds=sum(d['mac_foreground_seconds'] for d in included),
                           mac_unattributed_seconds=sum(d['mac_unattributed_seconds'] for d in included),
                           mac_app_only_seconds=sum(d['mac_app_only_seconds'] for d in included),
                           iphone_focus_seconds=sum(d['iphone_focus_seconds'] for d in included),
                           behavior=aggregate(included),days=included,
                           workstreams=[dict(label=label,status=status,sampled_seconds=seconds) for (label,status),seconds in streams.items()],
                           mac_apps=[],iphone_apps=[],local_commits=10,sum_daily_changed_files=20))
    year_streams=defaultdict(float)
    for d in history+days:
        for w in d['workstreams']:year_streams[(w['label'],w['status'])]+=w['sampled_seconds']
    year={**months[0], 'key':'2026','days_recorded':8,'behavior':aggregate(history+days),
          'workstreams':[dict(label=label,status=status,sampled_seconds=seconds)
                         for (label,status),seconds in year_streams.items()],
          'mac_foreground_seconds':sum(d['mac_foreground_seconds'] for d in history+days),
          'mac_active_seconds':sum(d['mac_active_seconds'] for d in history+days),
          'mac_unattributed_seconds':sum(d['mac_unattributed_seconds'] for d in history+days),
          'mac_app_only_seconds':sum(d['mac_app_only_seconds'] for d in history+days),
          'iphone_focus_seconds':sum(d['iphone_focus_seconds'] for d in history+days)}
    return dict(synthetic_fixture=True,generated_at_utc=now.isoformat(),timezone='America/Chicago',
                days=days,history=history,months=months,years=[year],week_behavior=aggregate(days),runtime={})
