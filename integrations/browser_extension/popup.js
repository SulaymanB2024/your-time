const HOST = 'com.yourtime.personal_activity_ledger';
const day = document.getElementById('day');
const rows = document.getElementById('rows');
const status = document.getElementById('status');
day.value = new Date().toLocaleDateString('sv-SE');
let offset = 0;
function message(payload) {
  return new Promise((resolve, reject) => chrome.runtime.sendNativeMessage(HOST, payload, reply => {
    const problem = chrome.runtime.lastError;
    if (problem) reject(new Error('The local review helper is unavailable.'));
    else if (!reply?.ok) reject(new Error('The local review request could not be completed.'));
    else resolve(reply);
  }));
}
function time(iso) { return new Date(iso).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'}); }
async function load() {
  rows.replaceChildren();status.textContent = 'Loading…';
  try {
    const result = await message({kind:'list_review',day:day.value,offset});
    status.textContent = result.total ? `${offset+1}–${Math.min(offset+20,result.total)} of ${result.total} review intervals` : 'No uncertain intervals need review.';
    for (const item of result.intervals) {
      const row = document.createElement('section');row.className='row';
      const top=document.createElement('div');top.className='top';
      const title=document.createElement('strong');title.textContent=`${time(item.start_utc)}–${time(item.end_utc)}`;
      const length=document.createElement('span');length.textContent=`${Math.round(item.unknown_seconds/60)}m unknown`;
      top.append(title,length);row.append(top);
      if(item.suggested_label){const hint=document.createElement('p');hint.className='hint';hint.textContent=`Screen suggests: ${item.suggested_label}`;row.append(hint);}
      const edit=document.createElement('div');edit.className='edit';
      const input=document.createElement('input');input.maxLength=65;input.setAttribute('aria-label',`Task label for ${title.textContent}`);input.placeholder='What were you working on?';input.value=item.confirmed_label||item.suggested_label||'';
      const save=document.createElement('button');save.type='button';save.textContent=item.confirmed_label?'Update':'Confirm';
      save.addEventListener('click',async()=>{save.disabled=true;try{await message({kind:'set_label',day:day.value,id:item.id,label:input.value});status.textContent='Task label saved locally.';save.textContent='Update';await load();}catch(error){status.textContent=error.message;}finally{save.disabled=false;}});
      edit.append(input,save);if(item.confirmed_label){const clear=document.createElement('button');clear.type='button';clear.textContent='Clear';clear.addEventListener('click',async()=>{clear.disabled=true;try{await message({kind:'clear_label',day:day.value,id:item.id});status.textContent='Label retracted locally.';input.value='';clear.remove();save.textContent='Confirm';await load();}catch(error){status.textContent=error.message;}finally{clear.disabled=false;}});edit.append(clear);}row.append(edit);if(item.confirmed_label){const result=document.createElement('button');result.className='result';result.type='button';result.textContent=item.outcome_marked?'Retract completed result':'Mark completed result';result.addEventListener('click',async()=>{if(input.value!==item.confirmed_label){status.textContent='Save the edited label first.';return;}result.disabled=true;try{await message({kind:item.outcome_marked?'retract_outcome':'mark_outcome',day:day.value,id:item.id});status.textContent=item.outcome_marked?'Result retracted locally.':'Result confirmed locally.';await load();}catch(error){status.textContent=error.message;}finally{result.disabled=false;}});row.append(result);}rows.append(row);
    }
    if(result.total>20){const nav=document.createElement('div');nav.className='tools';const previous=document.createElement('button');previous.textContent='Previous';previous.disabled=offset===0;previous.addEventListener('click',()=>{offset=Math.max(0,offset-20);load();});const next=document.createElement('button');next.textContent='Next';next.disabled=offset+20>=result.total;next.addEventListener('click',()=>{offset+=20;load();});nav.append(previous,next);rows.append(nav);}
  } catch(error) {status.textContent=error.message;}
}
document.getElementById('reload').addEventListener('click',load);
day.addEventListener('change',()=>{offset=0;load();});
load();
