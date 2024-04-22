#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:15:08 2024

@author: mactestviaguestaccount
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm



B1 = pd.read_csv("D:\\IMC 比赛\\round-5-island-data-bottle\\trades_round_1_day_-2_wn.csv", delimiter=';')
B2 = pd.read_csv("D:\\IMC 比赛\\round-5-island-data-bottle\\trades_round_1_day_-1_wn.csv", delimiter=';')
B2['timestamp'] = 1000000 + B2['timestamp']
B3 = pd.read_csv("D:\\IMC 比赛\\round-5-island-data-bottle\\trades_round_1_day_0_wn.csv", delimiter=';')
B3['timestamp'] = 2000000 + B3['timestamp']
df_B = pd.concat([B1, B2, B3])

participants=df_B['buyer'].unique()

characher_dict_buy={}
for buyer in df_B['buyer'].unique():
    characher_dict_buy[buyer]=df_B[df_B['buyer']==buyer].set_index('timestamp')
    
characher_dict_sell={}
for seller in df_B['seller'].unique():
    characher_dict_sell[seller]=df_B[df_B['seller']==seller].set_index('timestamp')
    


LOGS = pd.read_csv("D:\\IMC 比赛\\backtestV2\\prices_round_1_day_-2.csv", delimiter=',')
LOGS2 = pd.read_csv("D:\\IMC 比赛\\backtestV2\\prices_round_1_day_-1.csv", delimiter=',')
LOGS3 = pd.read_csv("D:\\IMC 比赛\\backtestV2\\prices_round_1_day_0.csv", delimiter=',')


df=pd.concat([LOGS,LOGS2,LOGS3])
LOGS=df
products_dict={}
for products in LOGS['product'].unique():
    
    products_dict[products]=LOGS[LOGS['product']==products].iloc[:,1:].set_index('timestamp')
    products_dict[products].index=range(0,3000000,100)
    #products_dict[products].index=range(0,100000,100)
    #if products == 'STARFRUIT':
     #   products_dict[products].index=products_dict['AMETHYSTS'].index
    
#%% STRAFRUITS

adam_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
adam_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Adam'][characher_dict_buy['Adam']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    adam_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Adam'][characher_dict_sell['Adam']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    adam_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

adam_df_starfruits.plot()
adam_df_starfruits['buy_orders'].dropna().plot()
adam_df_starfruits['sell_orders'].dropna().plot()

#ADAM, indicateur inverse pour le STAR??

valentina_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
valentina_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Valentina'][characher_dict_buy['Valentina']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    valentina_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Valentina'][characher_dict_sell['Valentina']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    valentina_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

valentina_df_starfruits.plot()
valentina_df_starfruits['buy_orders'].dropna().plot()
valentina_df_starfruits['sell_orders'].dropna().plot()
#Valentina = bon signal pour starfruit


ruby_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
ruby_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Ruby'][characher_dict_buy['Ruby']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    ruby_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Ruby'][characher_dict_sell['Ruby']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    ruby_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

ruby_df_starfruits.plot()
ruby_df_starfruits['buy_orders'].dropna().plot()
ruby_df_starfruits['sell_orders'].dropna().plot()
#Ruby Signal inverse

rhianna_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
rhianna_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Rhianna'][characher_dict_buy['Rhianna']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    rhianna_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Rhianna'][characher_dict_sell['Rhianna']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    rhianna_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

rhianna_df_starfruits.plot()
rhianna_df_starfruits['buy_orders'].dropna().plot()
rhianna_df_starfruits['sell_orders'].dropna().plot()
#rhianna : diffus

vinnie_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
vinnie_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Vinnie'][characher_dict_buy['Vinnie']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    vinnie_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Vinnie'][characher_dict_sell['Vinnie']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    vinnie_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

vinnie_df_starfruits.plot()
vinnie_df_starfruits['buy_orders'].dropna().plot()
vinnie_df_starfruits['sell_orders'].dropna().plot()
#Vinnie : plutot interessant

remy_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
remy_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Remy'][characher_dict_buy['Remy']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    remy_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Remy'][characher_dict_sell['Remy']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    remy_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

remy_df_starfruits.plot()
remy_df_starfruits['buy_orders'].dropna().plot()
remy_df_starfruits['sell_orders'].dropna().plot()
#Remy : indicateur inverse ou juste 0 pred?

vladimir_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
vladimir_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Vladimir'][characher_dict_buy['Vladimir']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    vladimir_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Vladimir'][characher_dict_sell['Vladimir']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    vladimir_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

vladimir_df_starfruits.plot()
vladimir_df_starfruits['buy_orders'].dropna().plot()
vladimir_df_starfruits['sell_orders'].dropna().plot()
#Vladimir : bon indicatur

amelia_df_starfruits=pd.DataFrame(columns=['mid_price','buy_orders','sell_orders'],index=products_dict['STARFRUIT'].index)
amelia_df_starfruits['mid_price']=products_dict['STARFRUIT']['mid_price']
temp=pd.DataFrame(columns=['buy_orders'])
temp['buy_orders']=characher_dict_buy['Amelia'][characher_dict_buy['Amelia']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    amelia_df_starfruits.loc[i,'buy_orders']=temp.loc[i,'buy_orders'].mean()
temp=pd.DataFrame(columns=['sell_orders'])
temp['sell_orders']=characher_dict_sell['Amelia'][characher_dict_sell['Amelia']['symbol']=='STARFRUIT']['price']
for i in temp.index:
    amelia_df_starfruits.loc[i,'sell_orders']=temp.loc[i,'sell_orders'].mean()

amelia_df_starfruits.plot()
amelia_df_starfruits['buy_orders'].dropna().plot()
amelia_df_starfruits['sell_orders'].dropna().plot()
#Amelia : bon indicateur


#%%

