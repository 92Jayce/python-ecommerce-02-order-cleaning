"""第02讲 清洗、去重与隔离问题记录
业务问题：怎样清洗金额、日期和渠道，又不悄悄丢掉问题？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 保留原始数据，统一文本格式
raw = pd.read_csv(DATA / 'orders_dirty.csv')
work = raw.copy()
for col in ['order_id','customer_id','status']:
    work[col] = work[col].astype('string').str.strip()
work['channel'] = work['channel'].astype('string').str.strip().str.lower().fillna('unknown')
duplicate_rows = work.loc[work.duplicated('order_id', keep='first')].copy()
work = work.drop_duplicates('order_id', keep='first').copy()

# %% 解析失败单独隔离，不用0代替未知金额
work['parsed_date'] = pd.to_datetime(work['order_date'],format='ISO8601',errors='coerce')
amount = work['paid_yuan_raw'].astype('string').str.replace(r'[¥,\s]','',regex=True)
work['parsed_yuan'] = pd.to_numeric(amount,errors='coerce')
bad_date = work['parsed_date'].isna()
bad_amount = work['parsed_yuan'].isna() | work['parsed_yuan'].lt(0)
work['issue'] = np.select([np.asarray(bad_date & bad_amount,dtype=bool),np.asarray(bad_date,dtype=bool),np.asarray(bad_amount,dtype=bool)],
    ['bad_date_and_amount','bad_date','bad_amount'],default='ok')
quarantine = work.loc[work['issue'].ne('ok')].copy()
clean = work.loc[work['issue'].eq('ok')].copy()
clean['parsed_cents'] = np.rint(clean['parsed_yuan']*100).astype('int64')
audit = pd.DataFrame({'stage':['raw','duplicates','quarantined_unique','clean_unique'],
    'rows':[len(raw),len(duplicate_rows),len(quarantine),len(clean)]})
show(audit); show(quarantine[['order_id','issue']])
save_table(clean,2,'cleaned_orders'); save_table(quarantine,2,'quarantine')
save_table(duplicate_rows,2,'duplicate_rows'); save_table(audit,2,'cleaning_audit')
assert len(raw)==len(duplicate_rows)+len(quarantine)+len(clean)
assert clean['parsed_cents'].eq(clean['paid_total_cents']).all()
print('这张脏表是专门的练习副本；后续课使用随包提供的标准数据，不把隔离表偷偷补成真值。')
