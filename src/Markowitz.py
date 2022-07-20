import pandas as pd
import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm #para las barras de progreso al lanzar simulaciones

def markowitz(assets, data, n_iteraciones):       
    '''
    Función para mediante la forntera eficiente asignar pesos a los tickers dados maximizando sharp ratio
    
    Parameters
    ----------
    assets : lista de tickers seleccionados
    data: dataframe con la ventana de datos a estudiar
    n_iteraciones: iteraciones para eficientar los pesos
         
    Returns
    -------
    allocations: DataFrame
        con los tickers 'tck' y los pesos 'alloc' asignados a los mismos
    '''
    log_returns = np.log(1+data.pct_change())

    port_returns = []
    port_vols = []

    #for i in tqdm(range(n_iteraciones)):
    for i in (range(n_iteraciones)):
        num_assets = len(assets)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) 
        port_ret = np.sum(log_returns.mean() * weights) * 252
        port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))       
        port_returns.append(port_ret)
        port_vols.append(port_var)

    def portfolio_stats(weights, log_returns):
        port_ret = np.sum(log_returns.mean() * weights) * 252
        port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe = port_ret/port_var    
        return {'Return': port_ret, 'Volatility': port_var, 'Sharpe': sharpe}

    def minimize_sharpe(weights, log_returns): 
        return -portfolio_stats(weights, log_returns)['Sharpe'] 

    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
    sharpe = port_returns/port_vols

    max_sr_vol = port_vols[sharpe.argmax()]
    max_sr_ret = port_returns[sharpe.argmax()]

    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((0,1) for x in range(num_assets))
    initializer = num_assets * [1./num_assets,]

    optimal_sharpe = optimize.minimize(minimize_sharpe, initializer, method = 'SLSQP', args = (log_returns,) ,bounds = bounds, constraints = constraints)
    optimal_sharpe_weights = optimal_sharpe['x'].round(4)
    optimal_stats = portfolio_stats(optimal_sharpe_weights, log_returns)
    
    #print("Pesos óptimos de la cartera: ", list(zip(assets, list(optimal_sharpe_weights*100))))
    #print("Retorno óptimo de la cartera: ", round(optimal_stats['Return']*100,4))
    #print("Volatilidad óptima de la cartera: ", round(optimal_stats['Volatility']*100,4))
    #print("Ratio Sharpe óptimo de la cartera: ", round(optimal_stats['Sharpe'],4))

    ret_op = round(optimal_stats['Return']*100,4)
    vol_op = round(optimal_stats['Volatility']*100,4)
    sharpe_op = round(optimal_stats['Sharpe'],4)
    
    allocations = pd.DataFrame({
        'tck': assets,
        'alloc': np.round(optimal_sharpe_weights,5)
    })
    allocations = allocations[(allocations.alloc >0)]
    allocations = allocations.sort_values(['alloc'], ascending=False)
    
    sum_alloc = allocations.alloc.sum()
    
    if (sum_alloc>0.999): 
        #print(allocations.iloc[0,1])
        allocations.iloc[0,1] =  allocations.iloc[0,1] + ((sum_alloc-1.001))
    allocations = allocations[(allocations.alloc >0)]
    lista = []
    lista = allocations.tck.tolist()
    return(allocations, lista, ret_op, vol_op, sharpe_op)