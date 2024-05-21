import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import create_engine
from scipy.stats import spearmanr
from sqlalchemy import text
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时'-'显示为方块的问题


# 本地算法库
from factor_functions import *


#性能检查
from line_profiler import LineProfiler
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner



def get_factor(factors, codes_list, start_time, end_time):
    """
    函数目的：从数据库中获取特定因子的数据。

    参数:
    factors: 包含你想要获取的因子的列表。例如：['ROC12','jumpdown_5']。
    codes_list: 包含你想要获取的股票代码的列表。例如：['000009.SZ','000010.SZ']。
    start_time: 你想要获取数据的开始日期，格式为 'yyyymmdd'。例如：'20000101'。
    end_time: 你想要获取数据的结束日期，格式为 'yyyymmdd'。例如：'20191231'。

    输出:
    df: 一个pandas DataFrame，其中包含了请求的数据。

    注意：因为这个函数需要从数据库中读取数据，所以可能需要一些时间来执行。建议使用成分股逐年的方式进行数据获取。
    """

    # 创建一个到数据库的连接
    engine = create_engine("mysql+pymysql://weizheng:yunpan123@124.220.177.115:3306/factordb")

    # 将股票代码和因子列表转换为 SQL 查询的一部分
    codes_sql = " , ".join(["'" + code + "'" for code in codes_list])
    factors_sql = " , ".join([" " + factor + " " for factor in factors])

    # 执行 SQL 查询并将结果读取为 pandas DataFrame
    df = pd.read_sql_query("SELECT S_INFO_WINDCODE , TRADE_DT ,"+factors_sql+" FROM FACTOR WHERE S_INFO_WINDCODE in ("+codes_sql+" ) and TRADE_DT> '"+start_time+"' and TRADE_DT< '"+end_time+"'", engine)

    return df



def ic_icir(factors,codes_list,start_time,end_time,shift):
    """
    函数目的：计算各因子的IC（Information Coefficient）和ICIR（Information Coefficient of Information Ratio）。

    参数:
    factors: 因子列表，例如：['ROC12','jumpdown_5']。
    codes_list: 股票代码列表，例如：['000009.SZ','000010.SZ']。
    start_time: 数据的开始日期，格式如：'20000101'。
    end_time: 数据的结束日期，格式如：'20191231'。
    shift: 计算IC时使用的收益率滞后期数。

    返回:
    df_ic_icir: 包含每个因子的IC和ICIR的DataFrame。
    """
    
    # 复制因子列表
    factors_ = factors.copy()
    factors_.append('S_DQ_PCTCHANGE')
    
    # 从数据源获取需要的数据
    df = get_factor(factors_,codes_list,start_time,end_time)
    
    # 确保数据是按日期排序的
    df = df.sort_values(by='TRADE_DT')
    
    # 计算每一只股票的下一个交易日的收盘价
    df['next_day_close'] = df.groupby('S_INFO_WINDCODE')['S_DQ_PCTCHANGE'].shift(-1*shift)
    df.dropna(inplace=True)
    
    # 创建空的DataFrame，用于存放结果
    df_ic_icir = pd.DataFrame(columns=['factor', 'ic_mean', 'icir'])

    plt.figure(figsize=(18, 6))  # 创建图形，并设置大小

    # 对于每个因子，计算其IC和ICIR
    for factor in factors_[:-1]:
        # 计算每个交易日的IC值
        ic_values = df.groupby('TRADE_DT').apply(lambda x: spearmanr(x[factor], x['next_day_close'])[0]).replace([np.inf, -np.inf], np.nan).dropna()
        
        # 转换日期为datetime格式，方便后续作图
        ic_values.index = pd.to_datetime(ic_values.index, format='%Y%m%d')  
        
        # 绘制IC的累计图
        ic_values.cumsum().plot(title="IC累计图",label=factor)  
        
        # 计算IC的均值和ICIR
        ic_mean = ic_values.mean()
        icir = ic_mean / ic_values.std()
        
        # 将结果添加到df_ic_icir中
        new_row = pd.DataFrame({'factor': factor, 'ic_mean': ic_mean, 'icir': icir}, index=[0])
        df_ic_icir = pd.concat([df_ic_icir,new_row ], join="inner")
        
    # 设置x轴的标签间隔，每间隔约一年显示一次标签
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=int(len(ic_values.index)/365))) 
    
    # 设置x轴标签的显示格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
    plt.legend()  # 显示图例
    plt.show()

    print("start_time:",start_time)
    print("end_time:",end_time)
    print("shift:",shift)
    print("codes_list_len:",len(codes_list))
    
    #将df_ic_icir的索引设置为factor
    df_ic_icir.set_index('factor',inplace=True)

    return df_ic_icir.sort_values(by='ic_mean',key=abs, ascending=False)

# @do_profile()
def write_to_sql(factor_df):
    """
    函数目的：将一个pandas DataFrame写入到MySQL数据库中。

    参数:
    factor_df: 一个pandas DataFrame，其中包含了需要写入数据库的数据。

    注意：这个函数假设了你已经有一个名为'FACTOR'的表，并且这个表的结构与factor_df的结构相匹配。
    如果FACTOR表中缺少factor_df中存在的列，这个函数将自动添加缺少的列。
    """

    # 创建数据库连接
    engine =create_engine("mysql+pymysql://weizheng:yunpan123@124.220.177.115:3306/factordb")
    # engine =create_engine("mysql+pymysql://root:123456asd@localhost:3306/factordb")
    # 将factor_df写入到名为'TMP'的临时表中
    factor_df.to_sql('TMP', con=engine, index=False, if_exists='replace')

    # 获取'FACTOR'表和'TMP'表的列名
    df_factor_colnames = pd.read_sql('SELECT * FROM FACTOR LIMIT 0', engine).columns.tolist()
    df_tmp_colnames = pd.read_sql('SELECT * FROM TMP LIMIT 0', engine).columns.tolist()

    # 查找'FACTOR'表中缺少的列
    diff_names = list(set(df_tmp_colnames) - set(df_factor_colnames))
    if len(diff_names) != 0:
        # 为'FACTOR'表添加缺少的列
        sql_1 = 'ALTER TABLE `FACTOR` '
        sql_2 = ",".join(["ADD COLUMN `"+diff_name+"` double NULL" for diff_name in diff_names])
        column_sql = sql_1 + sql_2
        with engine.begin() as connection:
            connection.execute(text(column_sql))

    # 将'TMP'表的内容写入到'FACTOR'表中
    # 如果'FACTOR'表中已经存在相同的记录（根据主键判断），则更新这个记录
    sql_1 = "INSERT INTO FACTOR ("+",".join(df_tmp_colnames)+") "
    sql_2 = "SELECT "+",".join(df_tmp_colnames)+" FROM TMP "
    sql_3 = "ON DUPLICATE KEY UPDATE "
    sql_4 = ",".join(["FACTOR."+df_tmp_colname+"=TMP."+df_tmp_colname for df_tmp_colname in df_tmp_colnames])
    with engine.begin() as connection:
        connection.execute(text(sql_1 + sql_2 + sql_3 + sql_4))
    
    # 删除临时表
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE TMP;"))
