import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = pd.read_csv('../Datasets/titanic/train.csv')
data_test  = pd.read_csv('../Datasets/titanic/test.csv')

# 利用pd返回数据的信息
# print(train_data.info())

'''
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
train_data.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title("获救情况 (1为获救)") # 标题
plt.ylabel("人数")

plt.subplot2grid((2,3),(0,1))
train_data.Pclass.value_counts().plot(kind="bar")
plt.ylabel("人数")
plt.title("乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(train_data.Survived, train_data.Age)
plt.ylabel("年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("按年龄看获救分布 (1为获救)")


plt.subplot2grid((2,3),(1,0), colspan=2)
train_data.Age[train_data.Pclass == 1].plot(kind='kde')
train_data.Age[train_data.Pclass == 2].plot(kind='kde')
train_data.Age[train_data.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")# plots an axis lable
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
train_data.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
plt.show()


#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = train_data.Pclass[train_data.Survived == 0].value_counts()
Survived_1 = train_data.Pclass[train_data.Survived == 1].value_counts()
df=pd.DataFrame({'获救':Survived_1, '未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("各乘客等级的获救情况")
plt.xlabel("乘客等级")
plt.ylabel("人数")
plt.show()
'''

# 看年龄
'''
Survived_Up = data_train.Survived[data_train.Age >= 25].value_counts()
Survived_Down = data_train.Survived[data_train.Age < 25].value_counts()
df=pd.DataFrame({u'高于25岁':Survived_Up, u'低于25岁':Survived_Down})
df.plot(kind='bar', stacked=True)
plt.title(u"按年龄看获救情况")
plt.xlabel(u"年龄")
plt.ylabel(u"人数")
plt.show()
'''


# 看看各性别的获救情况
'''
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数


Survived_m = data_train.Survived[train_data.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[train_data.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
print(df)
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()

'''

# 然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()
