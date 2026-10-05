"""Offline context exploration and calendar helpers for the dashboard.

Context presentation can be more specific than compact task rollups, but must
reconcile to the same identified time. No inferred label is treated as an outcome.
"""

CONTEXT_JS = r"""
function calendarShift(key,days){const d=new Date(key+'T12:00:00Z');d.setUTCDate(d.getUTCDate()+days);return d.toISOString().slice(0,10)}
function weekEndKeys(records){if(!records.length)return [];const keys=records.map(d=>d.day).sort(),ends=[];for(let key=keys[0];key<=keys.at(-1);key=calendarShift(key,1))ends.push(key);return ends}
function calendarWeek(records,end){const start=calendarShift(end,-6),present=new Map(records.map(d=>[d.day,d]));return Array.from({length:7},(_,i)=>{const day=calendarShift(start,i);return present.get(day)||{day,missing:true}})}
function contextKey(row){return row.status+'\u0000'+row.label}
function mergeContexts(rows){const totals=new Map();for(const row of rows){if(!(row.sampled_seconds>0))continue;const key=contextKey(row);if(!totals.has(key))totals.set(key,{label:row.label,status:row.status,sampled_seconds:0});totals.get(key).sampled_seconds+=row.sampled_seconds}return [...totals.values()].sort((a,b)=>b.sampled_seconds-a.sampled_seconds)}
function hasExactContexts(day){const sessions=day.behavior?.sessions;return !day.compact_only&&!!sessions?.length&&sessions.every(s=>s.label&&['specific_model','observed_window'].includes(s.status)&&s.sampled_seconds>0)&&Math.abs(sessions.reduce((sum,s)=>sum+s.sampled_seconds,0)-(day.mac_active_seconds||0))<=Math.max(.05,sessions.length*.0005+.002)}
function contextRows(day,exact=true){
 if(exact&&hasExactContexts(day))return mergeContexts(day.behavior.sessions);
 return mergeContexts(day.workstreams||[]);
}
function periodContexts(days,exact=true){return mergeContexts(days.flatMap(d=>contextRows(d,exact)))}
function aggregateContinuity(days){
 const b={identified_seconds:0,context_switches:0,session_count:0,longest_stretch_seconds:0,sustained_seconds:0,distribution:[]},distribution=new Map();
 for(const day of days){const row=day.behavior||{};for(const key of ['identified_seconds','context_switches','session_count','sustained_seconds'])b[key]+=row[key]||0;b.longest_stretch_seconds=Math.max(b.longest_stretch_seconds,row.longest_stretch_seconds||0);for(const item of row.distribution||[]){if(!distribution.has(item.label))distribution.set(item.label,{...item,sampled_seconds:0,sessions:0});const target=distribution.get(item.label);target.sampled_seconds+=item.sampled_seconds||0;target.sessions+=item.sessions||0}}
 b.distribution=[...distribution.values()];b.context_switches_per_identified_hour=b.identified_seconds?Math.round(b.context_switches*3600/b.identified_seconds*10)/10:null;return b;
}
function contextDaySeconds(day,row,exact=true){
 if(day.missing)return null;
 if(row.status==='unknown')return day.mac_unattributed_seconds||0;
 // A compact record cannot establish an exact window's absence.
 if(row.status==='observed_window'&&(!exact||!hasExactContexts(day)))return null;
 return contextRows(day,exact).filter(s=>contextKey(s)===contextKey(row)).reduce((total,s)=>total+s.sampled_seconds,0);
}
function foregroundRecords(day){const apps=new Map((day.behavior?.sessions||[]).map(r=>[r.start_utc+'\u0000'+contextKey(r),r.apps||[]]));return [...(day.timeline?.mac||[]).map(r=>({...r,device:'Mac',apps:apps.get(r.start_utc+'\u0000'+contextKey(r))||[]})),...(day.timeline?.iphone||[]).map(r=>({...r,device:'iPhone'}))].filter(r=>Date.parse(r.end_utc)>Date.parse(r.start_utc)).sort((a,b)=>Date.parse(a.start_utc)-Date.parse(b.start_utc))}
function contextDetailDays(days,records,exact){const recent=new Map(records.map(d=>[d.day,d]));return days.filter(d=>!d.missing).map(d=>exact?d:recent.get(d.day)||d)}
"""

INSPECTOR_JS = r"""
function selectedWeek(){return calendarWeek(dayRecords,$('week').value)}
function sourceName(status){return status==='specific_model'?'Estimated topic':status==='observed_window'?'Observed window':status==='unknown'?'Window details missing':status==='mixed'?'Grouped contexts':status==='phone'?'Synced app record':'Broad context'}
function preciseTime(seconds){const total=Math.max(0,Math.round(seconds||0)),h=Math.floor(total/3600),m=Math.floor(total%3600/60),s=total%60;return [h?h+'h':'',m?m+'m':'',s?s+'s':''].filter(Boolean).join(' ')||'0s'}
function chooseInterval(day,row){timelineSelection=day.day+'|'+row.device+'|'+row.start_utc+'|'+row.label;openDay(day.day);$('day-timeline').scrollIntoView({block:'center'})}
function intervalList(root,records,onSelect,deviceFilter=true){
 clear(root);if(!records.length){a(root,'p','quiet','No detailed intervals are available in this snapshot.');return}
 const controls=a(root,'div','interval-filters'),search=a(controls,'input','activity-search');search.type='search';search.placeholder='Search activity';search.setAttribute('aria-label','Search activity');
 let device=null;if(deviceFilter){device=a(controls,'select','activity-device');device.setAttribute('aria-label','Device');for(const label of ['All devices','Mac','iPhone']){const option=a(device,'option','',label);option.value=label}}
 const count=a(root,'p','quiet interval-count'),list=a(root,'div','interval-list');count.setAttribute('aria-live','polite');let limit=50;
 const draw=()=>{clear(list);const query=search.value.trim().toLocaleLowerCase(),shown=records.filter(r=>(!device||device.value==='All devices'||r.device===device.value)&&[r.label,...(r.apps||[])].join(' ').toLocaleLowerCase().includes(query));count.textContent=shown.length+' of '+records.length+' intervals · foreground time only';
  for(const row of shown.slice(0,limit)){const button=a(list,'button','interval-row');button.type='button';const time=a(button,'span','interval-time',(row.day?dayname(row.day.day)+' · ':'')+when(row.start_utc)+'–'+when(row.end_utc));const name=a(button,'span','interval-name');a(name,'span','interval-label',row.label);a(name,'span','interval-source',row.device+(row.device==='Mac'&&row.apps?.length?' · '+row.apps.join(' + '):'')+' · '+sourceName(row.device==='iPhone'?'phone':row.status));a(button,'span','interval-duration',preciseTime(row.sampled_seconds));button.addEventListener('click',()=>onSelect(row));button.setAttribute('aria-label',row.label+' · '+row.device+' · '+time.textContent+' · '+preciseTime(row.sampled_seconds))}
  if(!shown.length)a(list,'p','quiet','No matching intervals.');if(shown.length>limit){const more=a(list,'button','show-more','Show 50 more');more.type='button';more.addEventListener('click',()=>{limit+=50;draw()})}
 };search.addEventListener('input',()=>{limit=50;draw()});if(device)device.addEventListener('change',()=>{limit=50;draw()});draw();
}
function renderActivityLog(day){const root=$('activity-log');if(day.compact_only){clear(root);a(root,'p','quiet','This historical day retains totals. Detailed intervals are available for the latest seven days.');return}intervalList(root,foregroundRecords(day),row=>chooseInterval(day,row))}
function inspectContext(row,days,exact){
 const dialog=$('context-dialog'),body=$('context-body');clear(body);$('context-title').textContent=row.label;$('context-source').textContent=sourceName(row.status);$('context-total').textContent=preciseTime(row.sampled_seconds);
 const actual=days.filter(d=>!d.missing);a(body,'p','quiet',actual.length?days[0].day+' – '+days.at(-1).day+' · '+actual.length+' recorded '+(actual.length===1?'day':'days'):'No recorded days');
 const values=days.map(day=>({day,seconds:contextDaySeconds(day,row,exact)})),maximum=Math.max(1,...values.map(v=>v.seconds||0));const trend=a(body,'div','context-days');
 for(const value of values){const button=a(trend,'button','context-day');button.type='button';a(button,'span','',dayname(value.day.day));const track=a(button,'span','context-day-track');if(value.seconds!==null){const fill=a(track,'span','context-day-fill');fill.style.width=100*value.seconds/maximum+'%'}a(button,'span','context-day-value',value.seconds===null?(value.day.missing?'No record':'Detail unavailable'):preciseTime(value.seconds));button.disabled=!!value.day.missing;button.addEventListener('click',()=>{dialog.close();openDay(value.day.day)});button.setAttribute('aria-label',value.day.day+' · '+(value.seconds===null?'Detail unavailable':preciseTime(value.seconds)))}
 if(row.status==='specific_model')a(body,'p','context-note','The local model estimated this topic. Its duration comes from recorded Mac samples; it is not a confirmed accomplishment.');
 if(row.status==='observed_window')a(body,'p','context-note','This is the observed window label. It identifies context without assuming a task or result. Older compact records do not retain exact window detail.');
 if(!exact)a(body,'p','context-note','This breakdown uses the saved period totals. Detailed intervals below are a recent snapshot and may have newer labels or coverage.');
 const detailDays=contextDetailDays(days,dayRecords,exact),intervals=detailDays.flatMap(day=>foregroundRecords(day).filter(s=>s.device==='Mac'&&contextKey(s)===contextKey(row)).map(s=>({...s,day}))),detail=a(body,'details','context-intervals');a(detail,'summary','',intervals.length?'Inspect '+intervals.length+' recent intervals':'Interval availability');const list=a(detail,'div');intervalList(list,intervals,s=>{dialog.close();chooseInterval(s.day,s)},false);
 if(row.status==='unknown'){
  const candidates=detailDays.flatMap(day=>(day.review_candidates||[]).map(c=>({...c,day:day.day}))).sort((a,b)=>b.unknown_seconds-a.unknown_seconds);
  if(candidates.length){const review=a(body,'details','context-intervals');a(review,'summary','','Largest gaps to review');a(review,'p','context-note','These intervals lack a reliable window identity. Screen suggestions are unconfirmed and remain outside named task totals.');for(const c of candidates.slice(0,12)){const item=a(review,'div','gap-row');a(item,'span','',dayname(c.day)+' · '+when(c.start_utc)+'–'+when(c.end_utc));a(item,'strong','',preciseTime(c.unknown_seconds));if(c.suggested_label)a(item,'p','quiet','Screen suggestion: '+c.suggested_label)}}
 }
 dialog.showModal();
}
"""
