import pandas as pd
from matplotlib import pyplot as plt


csv0 = pd.read_csv('./dataset/datasetsporvo0.csv')
csv0 = pd.read_csv('./dataset/datasetsporvo0.csv')
csv1 = pd.read_csv('./dataset/datasetsporco.csv')
csv2 = pd.read_csv('./dataset/datasetsporco2.csv')
csv3 = pd.read_csv('./dataset/datasetsporco3.csv')
csv4 = pd.read_csv('./dataset/datasetsporco4.csv')
csv5 = pd.read_csv('./dataset/datasetsporco5.csv')
csv6 = pd.read_csv('./dataset/datasetsporco6.csv')
csv7 = pd.read_csv('./dataset/datasetsporco7.csv')

dataset = pd.concat([csv0, csv1, csv2, csv3, csv4, csv5, csv6, csv7], axis=0)

pos = dataset[(dataset['sentiment'] == 1) | (dataset['sentiment'] == 1.0)]
neg = dataset[(dataset['sentiment'] == -1) | (dataset['sentiment'] == -1.0)]
neu = dataset[(dataset['sentiment'] == 0) | (dataset['sentiment'] == 0.0)]

p = pos.groupby('date').size().reset_index(name='counts')
cum_p = p['counts'].cumsum()
n = neg.groupby('date').size().reset_index(name='counts')
cum_n = n['counts'].cumsum()
l = neu.groupby('date').size().reset_index(name='counts')
cum_l = l['counts'].cumsum()

#events_date = ['07-14-2021','07-22-2021', '08-06-2021', '08-09-2021', '09-01-2021', '09-07-2021', '09-17-2021', '09-22-2021', '10-09-2021', '10-12-2021','10-18-2021', '11-24-2021', '12-06-2021']

#plt.figure(figsize=(15, 8))
plt.axes().set_facecolor('white')
plt.plot(p['date'], cum_p, color='skyblue', linewidth=1.0, linestyle='--', label='positive')
plt.plot(n['date'], cum_n, color='green', linewidth=1.0, linestyle='--', label='negative')
plt.plot(l['date'], cum_l, color='black', linewidth=1.0, label='neutral')
#plt.xlim(['07-01-2021', '12-15-2021'])
#plt.xlabel('Events')
#plt.ylabel('Tweets')
plt.xticks([], rotation=90)
plt.legend()
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.title('Cumulated tweets')
plt.savefig('./img/cumulated_tweets.svg', format="svg")
plt.show()




