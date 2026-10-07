
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
