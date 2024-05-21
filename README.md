# gtja

factor_search：进行因子查询与回测，其中包含一些例子


cal_factors：用于更新因子。通过start_time、end_time、codes_list_all、function_list这四个变量选择需要更新的开始时间，结束时间，股票池，因子值


factor_functions：储存计算因子的函数


search_func：储存用于入库、查询、回测的辅助函数


pkl_fold：储存中间使用的到变量。例如某时刻的中证500成分股列表


