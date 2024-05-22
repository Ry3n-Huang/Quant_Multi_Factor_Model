import pandas as pd
import numpy as np
import datetime as dt
import sys




# from line_profiler import LineProfiler

# def do_profile(follow=[]):
#     def inner(func):
#         def profiled_func(*args, **kwargs):
#             try:
#                 profiler = LineProfiler()
#                 profiler.add_function(func)
#                 for f in follow:
#                     profiler.add_function(f)
#                 profiler.enable_by_count()
#                 return func(*args, **kwargs)
#             finally:
#                 profiler.print_stats()
#         return profiled_func
#     return inner



# # #读取某个数据

# #将df用pkl读取出来
# with open('df.pkl', 'rb') as f:
#     df_raw = pickle.load(f)
# # #另一个例子
# # engine1 = create_engine("mysql+pymysql://weizheng:yunpan123@124.220.177.115:3306/wind")
# # df = pd.read_sql_query("SELECT * FROM ASHAREEODPRICES WHERE S_INFO_WINDCODE in ('600605.SH' ) ", engine1)


"""
参数（下同）:
    df:
    一个dataframe，包含股票代码，日期和收盘价等信息

  `OBJECT_ID` varchar(100) NOT NULL COMMENT '对象ID',
  `S_INFO_WINDCODE` varchar(40) DEFAULT NULL COMMENT 'Wind代码',
  `TRADE_DT` varchar(8) DEFAULT NULL COMMENT '交易日期',
  `CRNCY_CODE` varchar(10) DEFAULT NULL COMMENT '货币代码',
  `S_DQ_PRECLOSE` decimal(20,4) DEFAULT NULL COMMENT '昨收盘价(元)',
  `S_DQ_OPEN` decimal(20,4) DEFAULT NULL COMMENT '开盘价(元)',
  `S_DQ_HIGH` decimal(20,4) DEFAULT NULL COMMENT '最高价(元)',
  `S_DQ_LOW` decimal(20,4) DEFAULT NULL COMMENT '最低价(元)',
  `S_DQ_CLOSE` decimal(20,4) DEFAULT NULL COMMENT '收盘价(元)',
  `S_DQ_CHANGE` decimal(20,4) DEFAULT NULL COMMENT '涨跌(元)',
  `S_DQ_PCTCHANGE` decimal(20,4) DEFAULT NULL COMMENT '涨跌幅(%)',
  `S_DQ_VOLUME` decimal(20,4) DEFAULT NULL COMMENT '成交量(手)',
  `S_DQ_AMOUNT` decimal(20,4) DEFAULT NULL COMMENT '成交金额(千元)',
  `S_DQ_ADJPRECLOSE` decimal(20,4) DEFAULT NULL COMMENT '复权昨收盘价(元)',
  `S_DQ_ADJOPEN` decimal(20,4) DEFAULT NULL COMMENT '复权开盘价(元)',
  `S_DQ_ADJHIGH` decimal(20,4) DEFAULT NULL COMMENT '复权最高价(元)',
  `S_DQ_ADJLOW` decimal(20,4) DEFAULT NULL COMMENT '复权最低价(元)',
  `S_DQ_ADJCLOSE` decimal(20,4) DEFAULT NULL COMMENT '复权收盘价(元)',
  `S_DQ_ADJFACTOR` decimal(20,6) DEFAULT NULL COMMENT '复权因子',
  `S_DQ_AVGPRICE` decimal(20,4) DEFAULT NULL COMMENT '均价(VWAP)',
  `S_DQ_TRADESTATUS` varchar(10) DEFAULT NULL COMMENT '交易状态',
  `S_DQ_TRADESTATUSCODE` decimal(5,0) DEFAULT NULL COMMENT '交易状态代码',
  `S_DQ_LIMIT` decimal(20,4) DEFAULT NULL COMMENT '涨停价(元)',
  `S_DQ_STOPPING` decimal(20,4) DEFAULT NULL COMMENT '跌停价(元)',
  `S_DQ_ADJCLOSE_BACKWARD` decimal(20,4) DEFAULT NULL COMMENT '前复权收盘价(元)',
  `OPDATE` datetime DEFAULT NULL,
  `OPMODE` varchar(1) DEFAULT NULL,


返回（下同）:
    df:
    一个dataframe，包含股票代码，日期和计算出来的因子
  

"""

def jumpdown(df):

    """计算jumpdown.

    公式：factor =RollingMean(op/cp_pre,S)，S=5,10,20

    市场逻辑：这个比率的滚动平均值可以反映出市场在一段时间内对股票开盘价的整体预期。
            如果这个比值的滚动平均值高于1，这可能意味着市场参与者普遍期待股票价格在开盘时上涨，或者说有更多的乐观情绪。
            反之，如果这个比值的滚动平均值低于1，那么市场可能对该股票持有更多的悲观预期。
    
    来源：
        国君研报《AI 投资方法论之二》

    """

    # # 首先，我们需要把日期字符串转换为日期对象
    # df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')

    # 计算新的列 'factor'
    df['factor'] = df['S_DQ_ADJOPEN'] / df['S_DQ_ADJPRECLOSE']

    # 按照股票代码和时间排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 分组并计算滚动平均值
    for s in [5, 10, 20]:
        df[f'jumpdown_{s}'] = df.groupby('S_INFO_WINDCODE')['factor'].transform(lambda x: x.rolling(s).mean())

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'jumpdown_5', 'jumpdown_10', 'jumpdown_20']]
    # df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def ROC12(df):
    """计算12日变动速率.

    公式：	①AX=今天的收盘价—12天前的收盘价
            ②BX=12天前的收盘价
            ③factor=AX/BX*100
    
    市场逻辑:
            12日变动速率是一种短期动量指标，它反映了股票价格在12个交易日内的变动幅度。
            12日变动速率越大，说明股票价格的变动幅度越大，市场趋势越强。
            12日变动速率越小，说明股票价格的变动幅度越小，市场趋势越弱。

    
    来源：
        聚宽因子库

    """
    # 首先，我们需要把日期字符串转换为日期对象
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')

    # 对数据进行排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算AX，BX和ROC
    df['AX'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x - x.shift(12))
    df['BX'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.shift(12))
    df['ROC12'] = df['AX'] / df['BX'] * 100

    # 选择需要的列，然后把日期转换为字符串
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'ROC12']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def wr_w(df):

    """计算威廉指数.

    公式：factor =(RollingMax(hp,N)-cp)/(RollingMax(hp,N)-RollingMin(lp,N))，N=5,10,20
    
    市场逻辑:用于衡量市场过度买入和过度卖出的情况
            当威廉指数大于80时，市场被认为是过度买入的,看跌.当威廉指数小于20时，市场被认为是过度卖出的,看多。

    来源：
        国君研报《AI 投资方法论之二》

    """
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])


    # 计算滚动窗口内的最高价和最低价
    for N in  [5,10,20]: # 假设你的窗口大小是20天
        df[f'RollingMax{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJHIGH'].rolling(window=N).max().reset_index(level=0, drop=True)
        df[f'RollingMin{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJLOW'].rolling(window=N).min().reset_index(level=0, drop=True)

        # 计算你的因子
        df[f'wr_w_{N}'] = (df[f'RollingMax{N}'] - df['S_DQ_ADJCLOSE']) / (df[f'RollingMax{N}'] - df[f'RollingMin{N}'])

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'wr_w_5', 'wr_w_10', 'wr_w_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)   
    return(df)


def rstr(df):

    """计算涨跌幅.

    公式：factor = PctChange(cp,N)

    市场逻辑:计算股票价格变化率，N=5,10,20
            如果 PctChange(cp, N) 大于某个阈值 threshold，则认为股票价格处于上涨趋势。
            如果 PctChange(cp, N) 小于某个负阈值 -threshold，则认为股票价格处于下跌趋势。
            如果 PctChange(cp, N) 在阈值范围内，可以认为股票价格处于相对稳定或震荡的状态。

    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    for n in n_values:
        df[f'rstr_{n}'] = (df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].diff(n) / df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].shift(n))


    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'rstr_5', 'rstr_10', 'rstr_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def stod(df):

    """计算涨跌幅.

    公式：factor = RollingMean(turn,S)

    市场逻辑:计算股票价格变化率，N=1,5,10,20
            如果 RollingMean(turn,S) 大于某个阈值 threshold，则认为股票价格处于上涨趋势。
            如果 RollingMean(turn,S) 小于某个负阈值 -threshold，则认为股票价格处于下跌趋势。
            如果 RollingMean(turn,S) 在阈值范围内，可以认为股票价格处于相对稳定或震荡的状态。

    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [1, 5, 10, 20]
    for n in n_values:
        df[f'stod_{n}'] = (df.groupby('S_INFO_WINDCODE')['S_DQ_PCTCHANGE'].rolling(n).mean()).reset_index(0, drop=True)

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'stod_1','stod_5', 'stod_10', 'stod_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def corr_cp_turnover(df):

    """计算涨跌幅.

    公式：factor = ollingCorr(cp,turn,N)

    市场逻辑:计算股票的复权昨收盘价(S_DQ_ADJPRECLOSE)与涨跌幅(S_DQ_PCTCHANGE)之间的滚动相关性。N=1,5,10,20
            相关性的值范围在-1和1之间，-1表示完全负相关，1表示完全正相关，0表示没有关系。
            这种计算的目的是为了检查历史价格（即复权昨收盘价）与当天的价格变动（即涨跌幅）之间的关系。
            正相关意味着当历史价格上升时，涨跌幅也倾向于上升，反之亦然。
            而负相关则意味着两者通常在相反的方向移动。

    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    for N in n_values:
        # 计算滚动相关性
        rolling_corr = df.groupby('S_INFO_WINDCODE')[['S_DQ_ADJPRECLOSE', 'S_DQ_PCTCHANGE']].rolling(window=N).corr().unstack()
        # 获取滚动相关性（这将是一个MultiIndex Series）
        rolling_corr = rolling_corr.iloc[:,1]
        # 将滚动相关性的索引重置为与df相同的索引
        rolling_corr = rolling_corr.reset_index(level=0, drop=True)
        # 添加新列到df
        df[f'corr_cp_turnover_{N}'] = rolling_corr

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT','corr_cp_turnover_5', 'corr_cp_turnover_10', 'corr_cp_turnover_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def vol(df):

    """计算涨跌幅.

    公式：factor = rollingStd(rtn,N) , N= 5,10,20,40

    市场逻辑:衡量和跟踪股票价格的稳定性和风险水平。N=5,10,20,40
            如果滚动标准差较大，说明这只股票的价格变动较大，风险较高；
            如果滚动标准差较小，说明这只股票的价格变动较小，相对稳定，风险较低。
    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20,40]
    for N in n_values:
        # 计算
        df[f'vol_{N}']  = df.groupby('S_INFO_WINDCODE')['S_DQ_PCTCHANGE'].rolling(window=N).std().reset_index(level=0, drop=True)

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'vol_5', 'vol_10', 'vol_20', 'vol_40']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def vstd(df):

    """.

    公式：factor = RollingStd(volume,N)/RollingMean(volume,N) 

    市场逻辑:用来评估市场的不确定性和交易活跃度的。N=5,10,20,40
            比值越高，说明成交量的波动相对于平均成交量来说较大，可能预示着市场的不确定性较高；
            比值越低，说明成交量的波动相对于平均成交量来说较小，市场可能相对较稳定。
    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    for N in n_values:
        # 计算
        df[f'vstd_{N}']  = df.groupby('S_INFO_WINDCODE')['S_DQ_VOLUME'].transform(lambda x: x.rolling(N).std() / x.rolling(N).mean())

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'vstd_5', 'vstd_10', 'vstd_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def volumerise(df):

    """计算涨跌幅.

    公式：factor = RollingMean(volume,N)/RollingMean(volume,60), N= 5,10,20,40

    市场逻辑:比较短期（N天）内的成交量平均值与较长期（60天）的成交量平均值。N=5,10,20,40
            在市场中，成交量往往被用作确认价格变动的有效性的重要指标。
            一般来说，当价格上涨时，如果伴随着成交量的增加，这通常被视为上涨趋势的确认，表明市场的买盘力量强大。
            反之，如果价格下跌，并且伴随着成交量的增加，这通常被视为下跌趋势的确认，表明市场的卖盘力量强大。
            当这个因子的值大于1时，表明短期内的成交量平均值超过了较长期的成交量平均值，可能暗示市场活跃度增强，可能预示着即将有重要的价格变动。
            当这个因子的值小于1时，表明短期内的成交量平均值低于较长期的成交量平均值，可能暗示市场活跃度较低，市场可能处于较为平静的状态。
    
    来源：
        国君研报《AI 投资方法论之二》

    """

    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])


    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    for N in n_values:
        # 计算
        df[f'volumerise_{N}']  = df.groupby('S_INFO_WINDCODE')['S_DQ_VOLUME'].transform(lambda x: x.rolling(N).mean() / x.rolling(60).mean())

    # 选择你需要的列
    df = df[['S_INFO_WINDCODE', 'TRADE_DT', 'volumerise_5', 'volumerise_10', 'volumerise_20']]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return(df)


def boll_down(df):
    """下轨线（布林线）指标

    公式：factor = (MA(CLOSE,M)-2*STD(CLOSE,M)) / 今日收盘价; M=5/10/20

    市场逻辑:下轨线（布林线）指标，N=5,10,20
            当价格触及下轨线时，意味着价格已经处于相对较低的位置，可能存在超卖（oversold）现象，是买入的潜在信号。
            然而，它并不是立即买入的信号，而是表明交易者需要观察其他市场信号，如价格反弹，或者其他指标显示超卖情况等。
            下轨线也常被用来设定止损点。如果价格跌破下轨线并持续下跌，可能意味着市场情绪发生了质的变化，交易者可能需要考虑卖出持仓。

    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
                
        df[f'MA_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.rolling(window=N).mean())
        df[f'STD_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.rolling(window=N).std())
        # 计算因子值
        df[f'{factor_name}_{N}'] = (df[f'MA_{N}'] - 2*df[f'STD_{N}']) / df['S_DQ_ADJCLOSE']
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def boll_up(df):
    """下轨线（布林线）指标

    公式：factor = (MA(CLOSE,M)+2*STD(CLOSE,M)) / 今日收盘价; M=5/10/20

    市场逻辑:下轨线（布林线）指标，N=5,10,20
            当价格触及上轨线时，意味着价格已经处于相对较高的位置，可能存在超买（oversold）现象，是卖出的潜在信号。
            然而，它并不是立即卖出的信号，而是表明交易者需要观察其他市场信号，或者其他指标显示超买情况等。
            上轨线也常被用来设定止盈点。如果价格跌破上轨线并持续上涨，可能意味着市场情绪发生了质的变化，交易者可能需要考虑买入持仓。

    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
                
        df[f'MA_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.rolling(window=N).mean())
        df[f'STD_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.rolling(window=N).std())
        # 计算因子值
        df[f'{factor_name}_{N}'] = (df[f'MA_{N}'] + 2*df[f'STD_{N}']) / df['S_DQ_ADJCLOSE']
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def arron_up(df):
    """Aroon指标上轨(该函数计算比较慢)

    公式：factor = Aroon(上升)=[(计算期天数-最高价后的天数)/计算期天数]*100; N=5/10/25

    市场逻辑:通过衡量某一期间内最高价和最低价出现时间的相对位置来判断市场的趋势变化和强度
            Aroon Up值越高，说明最高价出现的时间越接近现在，这可能是上涨趋势的一个信号。
            当Aroon Up值达到100%，说明在计算期间的最后一天，达到了最高价，这是一个强烈的上涨信号。
            当Aroon Up值下降，说明新的最高价出现的频率在减少，上涨动力可能在减弱。
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 25]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
                
        # 计算因子值
        df[f'{factor_name}_{N}'] = \
            df.groupby('S_INFO_WINDCODE')['S_DQ_HIGH'].\
            transform(lambda x: [(N - i) / N * 100 for i in (x.rolling(window=N).\
            apply(lambda y: y.argmax()))])
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def arron_down(df):
    """Aroon指标下轨(该函数计算比较慢)

    公式：factor = [(计算期天数-最低价后的天数)/计算期天数]*100; N=5/10/25

    市场逻辑:通过衡量某一期间内最高价和最低价出现时间的相对位置来判断市场的趋势变化和强度
            Aroon Down值越高，说明最低价出现的时间越接近现在，这可能是下跌趋势的一个信号。
            当Aroon Down值达到100%，说明在计算期间的最后一天，达到了最低价，这是一个强烈的下跌信号。
            当Aroon Down值下降，说明新的最低价出现的频率在减少，下跌动力可能在减弱。
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 25]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
                
        # 计算因子值
        df[f'{factor_name}_{N}'] = \
            df.groupby('S_INFO_WINDCODE')['S_DQ_HIGH'].\
            transform(lambda x: [(N - i) / N * 100 for i in (x.rolling(window=N).\
            apply(lambda y: y.argmin()))])
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def BIAS(df):
    """乖离率

    公式：factor = （收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100; N=5/10/25/60

    市场逻辑:测量市场价格与某个基准（通常是移动平均线）的偏离程度
            超买或超卖信号：乖离值过大可能意味着价格已经超过了其基本面价值，进入超买状态，反之则可能进入超卖状态。在这种情况下，投资者可能会考虑卖出或买入。
            趋势确认：乖离值可以帮助投资者确认市场趋势。正乖离值可能暗示价格上升趋势，而负乖离值可能暗示价格下降趋势。
            反转信号：当乖离值从正到负，或者从负到正变化时，可能预示着市场趋势的反转。
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5, 10, 20, 60]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
                
        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_CLOSE'].\
        transform(lambda x: (x - x.rolling(window=N).mean()) / x.rolling(window=N).mean() * 100)
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def returns(df):
    """收益率


    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    columns = ['S_INFO_WINDCODE', 'TRADE_DT','S_DQ_PCTCHANGE']
    
    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')


    return(df)


def VROC(df):
    """量变动速率指标

    公式：factor = 成交量减N日前的成交量，再除以N日前的成交量，放大100倍; N=12

    市场逻辑:成交量的变化与价格趋势的关系：
                通常情况下，成交量可以被认为是市场活跃度的一种体现。
                价格在上涨时，如果伴随着成交量的增加（即VROC为正），则表明更多的买方进入市场，这可能会继续推动价格上涨。
                相反，如果价格上涨，但成交量减少（即VROC为负），则可能表明上涨动力正在减弱，市场可能即将反转。

            成交量的突然变化可能预示着趋势变化：
                如果市场在平稳或下跌的过程中出现成交量的突然增加，可能预示着买方正在积极进场，这可能是市场趋势即将转为上涨的信号。
                同样，如果市场在上涨过程中出现成交量的突然减少，可能预示着卖方正在积极进场，这可能是市场趋势即将转为下跌的信号。

            成交量与价格之间的背离：
                如果市场价格持续上涨，但成交量持续下降（即VROC负值趋势），这被称为“背离”，可能预示着上涨趋势即将结束。
                同样，如果市场价格持续下跌，但成交量持续增加（即VROC正值趋势），这可能预示着下跌趋势即将结束。
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [12]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:

        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_VOLUME'].\
            apply(lambda x: ((x - x.shift(N)) / x.shift(N)) * 100).reset_index(0, drop=True)
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def TVMA(df):
    """成交金额的移动平均值

    公式：factor = N日成交金额的移动平均值; N=6/12/20

    市场逻辑:
    交易活跃度：
        成交金额可以被看作是市场交易活跃度的一个反映。
        如果过去6日的成交金额移动平均值上升，可能意味着更多的资金正在进入市场，这可能预示着市场活动度提升、买卖双方交易更加活跃。
    市场趋势判断：
        如果成交金额的移动平均值呈上升趋势，这可能意味着市场买盘强势，可能会推动价格上涨；
        相反，如果移动平均值呈下降趋势，可能意味着市场买盘弱势，可能会导致价格下跌。
    信号识别：
        在某些情况下，成交金额的移动平均线可能会与价格产生某种关系，如：
        价格突破成交金额的移动平均线可能会被视为买入信号，反之，可能会被视为卖出信号。
    异动检测：
        如果某一天的成交金额远超过其移动平均值，可能意味着市场出现了异常情况（如大量买入或卖出）。
        这可能是因为某些信息（如公司的重大消息）导致市场参与者的交易行为发生改变。
    
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [6,12,20]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
 
        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_AMOUNT']\
                                    .transform(lambda x: x.rolling(N).mean())
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def VEMA(df):
    """成交量的N日指数移动平均

    公式：factor = 成交量的N日指数移动平均; N=5/10/20

    市场逻辑:
        在交易量上升的情况下，如果股价也在上涨，那么这通常意味着市场对于股票的需求在增加，对于股票的看涨情绪可能在加强。EMA的上升可能预示着这种趋势的持续。
        在交易量上升的情况下，如果股价在下跌，那么这可能表示市场对于股票的看空情绪增强，短期内股票价格可能会继续下跌。
        如果交易量在下降，无论股价是上涨还是下跌，都可能表示市场对于这只股票的兴趣在下降，市场动力可能在减弱。
    
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [5,10,12,26]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']
    for N in n_values:
 
        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_VOLUME'].transform(lambda x: x.ewm(span=N).mean())
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def BR(df):
    """意愿指标

    公式：factor = N日内（当日最高价－昨日收盘价）之和 / N日内（昨日收盘价－当日最低价）之和×100; N=26

    市场逻辑:
        当这个比率较高时，说明在过去的N日内，股票的上涨力度较强，买盘相对活跃。
        这可能暗示了投资者对该股票的前景比较看好，或者市场上存在较强的买入动力。
        因此，这可能是一个看涨的信号。
        当这个比率较低时，说明在过去的N日内，股票的下跌力度较强，卖盘相对活跃。
        这可能表明投资者对该股票的前景比较悲观，或者市场上存在较强的卖出动力。
        因此，这可能是一个看跌的信号。
    
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [26]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']

    # 计算每只股票每日的价格差
    df['high_diff'] = df['S_DQ_ADJHIGH'] - df['S_DQ_PRECLOSE']
    df['low_diff'] = df['S_DQ_ADJPRECLOSE'] - df['S_DQ_ADJLOW']


    for N in n_values:
        
        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['high_diff'].rolling(window=N).sum().reset_index(0, drop=True)/ \
                                   df.groupby('S_INFO_WINDCODE')['low_diff'].rolling(window=N).sum().reset_index(0, drop=True)
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def VMACD(df):
    """成交量指数平滑异同移动平均线

    公式：factor = 快的指数移动平均线（EMA12）减去慢的指数移动平均线（EMA26）得到快线DIFF, 由DIFF的M日移动平均得到DEA，由DIFF-DEA的值得到MACD

    市场逻辑:

    
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',f'{factor_name}']
    df['VOLUME_EMA_SHORT'] = df['S_DQ_VOLUME'].ewm(span=12).mean()
    df['VOLUME_EMA_LONG'] = df['S_DQ_VOLUME'].ewm(span=26).mean()
    df['VDIFF'] = df['VOLUME_EMA_SHORT'] - df['VOLUME_EMA_LONG']
    df['VDEA'] =  df['VDIFF'].ewm(span=9).mean()
    df['VMACD'] = df['VDIFF'] - df['VDEA']
    
    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def ATR(df):
    """N日均幅指标

    公式：factor = 
        "真实振幅"是一种在技术分析中使用的指标，它反映了一只股票在一个交易日内的价格波动程度。它被定义为以下三者中的最大值：
        当日最高价和最低价之间的差价（绝对值）
        当日最高价和前一日收盘价之间的差价（绝对值）
        当日最低价和前一日收盘价之间的差价（绝对值）

    市场逻辑:
        真实振幅和它的移动平均可以反映市场的波动程度，
        如果真实振幅的14日移动平均上升，说明市场波动加大，可能存在大的价格变动，也可能市场存在较大的信息不对称，此时投资者应对市场进行关注。
        反之，如果真实振幅的14日移动平均下降，说明市场相对稳定，可能是市场正在等待某些信息的发布或者是市场较为平静。
    
    来源：
        聚宽因子库

    """
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    n_values = [6,14]
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']

    df['TR'] = df[['S_DQ_HIGH', 'S_DQ_ADJPRECLOSE']].max(axis=1) - df[['S_DQ_LOW', 'S_DQ_ADJPRECLOSE']].min(axis=1)

    for N in n_values:
        # 计算因子值
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['TR']\
            .rolling(window=N).mean().reset_index(0, drop=True)
        columns.append(f'{factor_name}_{N}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)





def calu0116(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    n_values = [5, 10, 20]

    df[f'HighMinusOpen'] = df['S_DQ_HIGH'] - df['S_DQ_OPEN']
    df[f'OpenMinusLow'] = df['S_DQ_OPEN'] - df['S_DQ_LOW']

    for n in n_values:
        # 计算分子：sum(high - open, n)

        df[f'HighMinusOpen_{n}'] = df.groupby('S_INFO_WINDCODE')[f'HighMinusOpen'].rolling(window=n).sum().reset_index(0, drop=True)

        # 计算分母：sum(open - low, n)

        df[f'OpenMinusLow_{n}'] = df.groupby('S_INFO_WINDCODE')[f'OpenMinusLow'].rolling(window=n).sum().reset_index(0, drop=True)

        # 计算指标：factor
        df[f'{factor_name}_{n}'] = df[f'HighMinusOpen_{n}'] / df[f'OpenMinusLow_{n}'] * 100

        
        columns.append(f'{factor_name}_{n}')

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)



def calu0120(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    
    columns.append(f'{factor_name}')
    df['52WeekHigh'] = df.groupby('S_INFO_WINDCODE')['S_DQ_HIGH'].rolling(window=252).max().reset_index(0, drop=True)
    df[f'{factor_name}'] = df['S_DQ_CLOSE'] / df['52WeekHigh']

    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0124(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    # 计算A、B、C、D、E、F、G的值
    df['A'] = np.abs(df['S_DQ_HIGH'] - df['S_DQ_CLOSE'].shift(1))
    df['B'] = np.abs(df['S_DQ_LOW'] - df['S_DQ_CLOSE'].shift(1))
    df['C'] = np.abs(df['S_DQ_HIGH'] - df['S_DQ_LOW'].shift(1))
    df['D'] = np.abs(df['S_DQ_CLOSE'].shift(1) - df['S_DQ_OPEN'].shift(1))
    df['E'] = df['S_DQ_CLOSE'] - df['S_DQ_CLOSE'].shift(1)
    df['F'] = df['S_DQ_CLOSE'] - df['S_DQ_OPEN']
    df['G'] = df['S_DQ_CLOSE'].shift(1) - df['S_DQ_OPEN'].shift(1)

    # 计算X、K、R、SI的值
    df['X'] = df['E'] + 0.5 * df['F'] + df['G']
    df['K'] = np.maximum(df['A'], df['B'])
    df['R'] = np.where((df['A'] > df['B']) & (df['A'] > df['C']),
                    df['A'] + 0.5 * df['B'] + 0.25 * df['D'],
                    np.where((df['B'] > df['A']) & (df['B'] > df['C']),
                                df['B'] + 0.5 * df['A'] + 0.25 * df['D'],
                                df['C'] + 0.25 * df['D']))
    df['SI'] = 16 * df['X'] / (df['R'] * df['K'])


    # 计算指标
    n_values = [5, 10, 20]
    # 计算ASI(N)
    for n in n_values:

        # 计算指标：factor
        df[f'{factor_name}_{n}'] = df['SI'].rolling(window=n).sum()

        # 将指标添加到columns中
        columns.append(f'{factor_name}_{n}')


    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)



def calu0129(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    # 计算CLV的值
    df['CLV'] = df['S_DQ_VOLUME'] * ((df['S_DQ_CLOSE'] - df['S_DQ_LOW']) - (df['S_DQ_HIGH'] - df['S_DQ_CLOSE'])) / (df['S_DQ_HIGH'] - df['S_DQ_CLOSE'])


    # 计算指标
    n_values = [5, 10, 20]
    # 计算ASI(N)
    for n in n_values:

        # 计算指标：factor
        df[f'{factor_name}_{n}'] = df['CLV'].rolling(window=n).sum() / df['S_DQ_VOLUME'].rolling(window=n).sum() * 100

        # 将指标添加到columns中
        columns.append(f'{factor_name}_{n}')


    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0131(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将交易日期转换为datetime格式
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']




    # 计算指标
    n_values = [1,2,3]
    # 计算ASI(N)
    for n in n_values:
        peried = n*20
        # 计算指标：factor
        df[f'{factor_name}_{n}'] = df['S_DQ_PCTCHANGE'].rolling(window=peried).sum()

        # 将指标添加到columns中
        columns.append(f'{factor_name}_{n}')


    # 选择需要的列
    df = df[columns]
    df['TRADE_DT'] = df['TRADE_DT'].dt.strftime('%Y%m%d')
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0205(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']



    # 计算指标
    n_values = [21,42,63]
    # 计算ASI(N)
    for N in n_values:

        # 计算指标：factor
        df[f'{factor_name}_{N}'] = df.groupby('S_INFO_WINDCODE')['S_DQ_PCTCHANGE'].rolling(N).var().values

        # 将指标添加到columns中
        columns.append(f'{factor_name}_{N}')


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)



def calu0211(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    df['HIGH_LOW'] = df['S_DQ_ADJHIGH'] - df['S_DQ_ADJLOW']


    for N in [5, 10, 20]:
        # 计算EMA
        EMA_col_name = f'EMA_HIGH_LOW_{N}'
        df[EMA_col_name] = df.groupby('S_INFO_WINDCODE')['HIGH_LOW'].transform(lambda x: x.ewm(span=N, adjust=False).mean())

        # 移位EMA
        EMA_shifted_col_name = f'{EMA_col_name}_shifted'
        df[EMA_shifted_col_name] = df.groupby('S_INFO_WINDCODE')[EMA_col_name].shift(N)       

        df[f'{factor_name}_{N}'] = (df[EMA_col_name] - df[EMA_shifted_col_name]) / df[EMA_col_name] * 100
        # 将指标添加到columns中
        columns.append(f'{factor_name}_{N}')


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0218(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT']


    df['CLOSE_prev'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].shift(1)
    df['HIGH_MINUS_CLOSE_prev'] = df['S_DQ_ADJHIGH'] - df['CLOSE_prev']
    df['CLOSE_prev_MINUS_LOW'] = df['CLOSE_prev'] - df['S_DQ_ADJLOW']
    df['MAX_HIGH_MINUS_CLOSE_prev'] = df['HIGH_MINUS_CLOSE_prev'].apply(lambda x: max(x, 0))
    df['MAX_CLOSE_prev_MINUS_LOW'] = df['CLOSE_prev_MINUS_LOW'].apply(lambda x: max(x, 0))


    for N in [5, 10, 20]:
        df[f'SUM_MAX_HIGH_MINUS_CLOSE_prev_{N}'] = df.groupby('S_INFO_WINDCODE')['MAX_HIGH_MINUS_CLOSE_prev'].transform(lambda x: x.rolling(N).sum())
        df[f'SUM_MAX_CLOSE_prev_MINUS_LOW_{N}'] = df.groupby('S_INFO_WINDCODE')['MAX_CLOSE_prev_MINUS_LOW'].transform(lambda x: x.rolling(N).sum())
        df[f'{factor_name}_{N}'] = df[f'SUM_MAX_HIGH_MINUS_CLOSE_prev_{N}'] / df[f'SUM_MAX_CLOSE_prev_MINUS_LOW_{N}']
        
        # 将指标添加到columns中
        columns.append(f'{factor_name}_{N}')


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0225(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]

    # 设置EMA的周期
    N1 = 12
    N2 = 26
    N3 = 9

    # 计算两个EMA
    df['EMA_CLOSE_N1'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.ewm(span=N1).mean())
    df['EMA_CLOSE_N2'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].transform(lambda x: x.ewm(span=N2).mean())

    # 计算DIF
    df['DIF'] = df['EMA_CLOSE_N1'] - df['EMA_CLOSE_N2']

    # 计算DEA
    df['DEA'] = df.groupby('S_INFO_WINDCODE')['DIF'].transform(lambda x: x.ewm(span=N3).mean())

    # 计算MACD值
    df[factor_name] = 2 * (df['DIF'] - df['DEA'])


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)



def calu0305(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]
    # 设置N1和N2
    # 设置N1和N2
    N1 = 34
    N2 = 55
    # 定义计算函数
    def calculate_kvo(group):
        group['TR'] = np.where(group['S_DQ_ADJHIGH'] + group['S_DQ_ADJLOW'] + group['S_DQ_ADJCLOSE'] > group['S_DQ_ADJHIGH'].shift(1) + group['S_DQ_ADJLOW'].shift(1) + group['S_DQ_ADJCLOSE'].shift(1), 1, -1)
        group['DM'] = group['S_DQ_ADJHIGH'] - group['S_DQ_ADJLOW']
        group['CM'] = group['DM'].copy() # 初始化CM
        group.loc[group['TR'] == group['TR'].shift(1), 'CM'] = group['CM'].shift(1) + group['DM'] # 更新CM
        group['VF'] = group['S_DQ_VOLUME'] * np.abs(2 * (group['DM']/group['CM'] - 1)) * group['TR'] * 100
        group['EMA_VF_N1'] = group['VF'].ewm(span=N1, adjust=False).mean().shift(N1)
        group['EMA_VF_N2'] = group['VF'].ewm(span=N2, adjust=False).mean().shift(N2)
        group[factor_name] = group['EMA_VF_N1'] - group['EMA_VF_N2']
        return group

    # 对每只股票独立进行计算
    df = df.groupby('S_INFO_WINDCODE').apply(calculate_kvo).reset_index(drop=True)

    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0315(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]
    N = 20

    # 计算DMZ和DMF
    df['DMZ'] = np.where(df['S_DQ_ADJHIGH'] + df['S_DQ_ADJLOW'] <= df['S_DQ_ADJHIGH'].shift(1) + df['S_DQ_ADJLOW'].shift(1), 0, 
                        np.maximum(np.abs(df['S_DQ_ADJHIGH'] - df['S_DQ_ADJHIGH'].shift(1)), 
                                    np.abs(df['S_DQ_ADJLOW'] - df['S_DQ_ADJLOW'].shift(1))))

    df['DMF'] = np.where(df['S_DQ_ADJHIGH'] + df['S_DQ_ADJLOW'] > df['S_DQ_ADJHIGH'].shift(1) + df['S_DQ_ADJLOW'].shift(1), 0, 
                        np.maximum(np.abs(df['S_DQ_ADJHIGH'] - df['S_DQ_ADJHIGH'].shift(1)), 
                                    np.abs(df['S_DQ_ADJLOW'] - df['S_DQ_ADJLOW'].shift(1))))

    # 分组计算
    grouped = df.groupby('S_INFO_WINDCODE')

    # 计算DIZ和DIF
    df['DIZ'] = grouped.apply(lambda x: x['DMZ'].rolling(N).sum() / (x['DMZ'].rolling(N).sum() + x['DMF'].rolling(N).sum()) * 100).reset_index(level=0, drop=True)
    df['DIF'] = grouped.apply(lambda x: x['DMF'].rolling(N).sum() / (x['DMZ'].rolling(N).sum() + x['DMF'].rolling(N).sum()) * 100).reset_index(level=0, drop=True)

    # 当分母为0时，将DIZ和DIF设为0
    df['DIZ'].fillna(0, inplace=True)
    df['DIF'].fillna(0, inplace=True)



    # 对每只股票独立进行计算
    df[factor_name] = df['DIZ'] - df['DIF']


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0321(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]


    # 定义滚动窗口的大小
    N = 20

    # 将数据分组，对每一只股票进行处理
    grouped = df.groupby('S_INFO_WINDCODE')

    # 定义两个函数，分别用于计算阿隆上升和下降指标
    def calculate_aroon_up(series):
        return 100 * (N - series[::-1].idxmax()) / N

    def calculate_aroon_down(series):
        return 100 * (N - series[::-1].idxmin()) / N

    # 对每一只股票的收盘价序列应用上述函数
    df['Aroon_Up'] = grouped['S_DQ_ADJCLOSE'].rolling(N).apply(calculate_aroon_up, raw=False).reset_index(level=0, drop=True)
    df['Aroon_Down'] = grouped['S_DQ_ADJCLOSE'].rolling(N).apply(calculate_aroon_down, raw=False).reset_index(level=0, drop=True)

    # 计算阿隆振荡指数
    df[factor_name] = df['Aroon_Up'] - df['Aroon_Down']


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)



def calu0413(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]

    # 定义参数
    N1 = 20
    N2 = 5

    # 对数据按照股票代码进行分组
    grouped = df.groupby('S_INFO_WINDCODE')

    # 计算TR
    df['CLOSE_1'] = grouped['S_DQ_ADJCLOSE'].shift(1)
    df['TR'] = df[['S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'CLOSE_1']].apply(lambda x: max(x[0] - x[1], abs(x[2] - x[0]), abs(x[2] - x[1])), axis=1).reset_index(level=0, drop=True)

    # 计算W
    df['W'] = np.where(df['S_DQ_ADJCLOSE'] > df['CLOSE_1'], df['TR'] / (df['S_DQ_ADJCLOSE'] - df['CLOSE_1']), df['TR'])

    # 计算SR
    df['W_MAX'] = grouped['W'].rolling(N1).max().reset_index(level=0, drop=True)
    df['W_MIN'] = grouped['W'].rolling(N1).min().reset_index(level=0, drop=True)
    df['SR'] = np.where(df['W_MAX'] - df['W_MIN'] > 0, (df['W'] - df['W_MIN']) / (df['W_MAX'] - df['W_MIN']) * 100, (df['W'] - df['W_MIN']) * 100)

    # 计算RI
    df[factor_name] = df.groupby('S_INFO_WINDCODE')['SR'].transform(lambda x: x.ewm(span=N2).mean())


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


def calu0427(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]

    # 首先计算前一天的收盘价
    df['PREV_CLOSE'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].shift(1)

    # 对每一只股票进行分组处理
    def calculate_obv(group):
        # 初始OBV设为0
        obv = [0]
        for i in range(1, len(group)):
            if group['S_DQ_ADJCLOSE'].iloc[i] > group['PREV_CLOSE'].iloc[i]:
                obv.append(obv[-1] + group['S_DQ_VOLUME'].iloc[i])
            elif group['S_DQ_ADJCLOSE'].iloc[i] < group['PREV_CLOSE'].iloc[i]:
                obv.append(obv[-1] - group['S_DQ_VOLUME'].iloc[i])
            else:
                obv.append(obv[-1])
        group[factor_name] = obv
        return group

    # 对每一只股票分别计算OBV
    df = df.groupby('S_INFO_WINDCODE').apply(calculate_obv).reset_index(drop=True)

    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)

def calu0501(df):
    # 获取函数名
    factor_name = sys._getframe().f_code.co_name
    # 将数据按股票代码和日期排序
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    # 计算每只股票的factor
    columns = ['S_INFO_WINDCODE', 'TRADE_DT',factor_name]


    # 首先计算前一天的典型价格
    df['TP'] = (df['S_DQ_ADJHIGH'] + df['S_DQ_ADJLOW'] + df['S_DQ_ADJCLOSE']) / 3
    df['PREV_TP'] = df.groupby('S_INFO_WINDCODE')['TP'].shift(1)

    # 计算Money Flow
    df['MF'] = df['TP'] * df['S_DQ_VOLUME']

    # 设定滑动窗口大小
    N = 20

    # 计算正向和负向的资金流
    df['PF'] = df[df['TP'] > df['PREV_TP']].groupby('S_INFO_WINDCODE')['MF'].rolling(N).sum().reset_index(0, drop=True)
    df['NF'] = df[df['TP'] <= df['PREV_TP']].groupby('S_INFO_WINDCODE')['MF'].rolling(N).sum().reset_index(0, drop=True)

    # 用fill na处理空值
    df['PF'].fillna(0, inplace=True)
    df['NF'].fillna(0, inplace=True)

    # 计算资金比率
    df['MR'] = df['PF'] / df['NF']

    # 计算MFI
    df[factor_name] = 100 - (100 / (1 + df['MR']))


    # 选择需要的列
    df = df[columns]
    # 将inf替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return(df)


#Weihang Huang begins here::

