import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from numba import jit, prange
import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spy_option_pricing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SPYDataHandler:
    """
    Processes SPY data files for use in option pricing models
    """
    
    def __init__(self, base_path: str):
        """
        Initialize with path to SPY data files
        
        Parameters:
        - base_path: Base directory for SPY data files
        """
        self.base_path = base_path
        self.logger = logger
        
    def load_data(self, quarter: str, month: str) -> pd.DataFrame:
        """
        Load SPY data for a specific quarter and month
        
        Parameters:
        - quarter: Quarter identifier (e.g., 'q1')
        - month: Month identifier (e.g., '01' for January)
        
        Returns:
        - Processed DataFrame with SPY option data
        """
        try:
            file_path = os.path.join(
                self.base_path, 
                f"spy_eod_2023{quarter}-zfoivd/spy_eod_2023{month}.txt"
            )
            
            self.logger.info(f"Loading SPY data from: {file_path}")
            
            # Load raw data with correct delimiter
            raw_data = pd.read_csv(file_path, delimiter=", ")
            
            self.logger.info(f"Loaded {len(raw_data)} records from {month}/2023")
            return raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading SPY data: {e}")
            raise
            
    def process_data(self, raw_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process raw SPY data into separate calls and puts DataFrames
        
        Parameters:
        - raw_data: Raw SPY data DataFrame
        
        Returns:
        - Dictionary with processed DataFrames for calls and puts
        """
        # Define columns to drop for calls and puts
        call_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", 
                          "[EXPIRE_UNIX]", "[P_BID]", "[P_ASK]", "[P_SIZE]", "[P_LAST]", 
                          "[P_DELTA]", "[P_GAMMA]", "[P_VEGA]", "[P_THETA]", "[P_RHO]", 
                          "[P_IV]", "[P_VOLUME]"]
        
        put_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", 
                         "[EXPIRE_UNIX]", "[C_DELTA]", "[C_GAMMA]", "[C_VEGA]", 
                         "[C_THETA]", "[C_RHO]", "[C_IV]", "[C_VOLUME]", "[C_LAST]", 
                         "[C_SIZE]", "[C_BID]", "[C_ASK]"]
        
        # Create separate DataFrames for calls and puts
        df_calls = raw_data.drop(columns=call_dropables)
        df_puts = raw_data.drop(columns=put_dropables)
        
        # Extract unique underlying prices by quote readtime
        timeser_underlying = raw_data.groupby("[QUOTE_READTIME]")["[UNDERLYING_LAST]"].last().reset_index()
        
        return {
            "calls": df_calls,
            "puts": df_puts,
            "underlying": timeser_underlying
        }
    
    def extract_market_parameters(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract key market parameters from processed SPY data
        
        Parameters:
        - data: Processed data dictionary
        
        Returns:
        - Dictionary with market parameters
        """
        calls_df = data["calls"]
        puts_df = data["puts"]
        underlying_df = data["underlying"]
        
        # Extract current underlying price
        current_price = underlying_df["[UNDERLYING_LAST]"].iloc[-1]
        
        # Calculate historical volatility (annualized)
        price_series = underlying_df["[UNDERLYING_LAST]"]
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        historical_vol = log_returns.std() * np.sqrt(252)
        
        # Extract implied volatility from ATM options
        # Find ATM options (strike price closest to current price)
        atm_calls = calls_df[abs(calls_df["[STRIKE]"] - current_price) < 1]
        if len(atm_calls) > 0:
            avg_iv_calls = atm_calls["[C_IV]"].mean()
        else:
            avg_iv_calls = None
            
        atm_puts = puts_df[abs(puts_df["[STRIKE]"] - current_price) < 1]
        if len(atm_puts) > 0:
            avg_iv_puts = atm_puts["[P_IV]"].mean()
        else:
            avg_iv_puts = None
            
        # Use average of call and put IV if available
        if avg_iv_calls is not None and avg_iv_puts is not None:
            implied_vol = (avg_iv_calls + avg_iv_puts) / 2
        elif avg_iv_calls is not None:
            implied_vol = avg_iv_calls
        elif avg_iv_puts is not None:
            implied_vol = avg_iv_puts
        else:
            implied_vol = historical_vol
            
        # Extract DTE (days to expiration)
        avg_dte = calls_df["[DTE]"].iloc[-1]  # Use latest DTE
        
        # Extract dividend yield (placeholder - would use actual div yield in production)
        div_yield = 0.015  # 1.5% placeholder for SPY
        
        # Extract risk-free rate (placeholder - would use Treasury yield in production)
        risk_free_rate = 0.04  # 4% placeholder
        
        return {
            "spot_price": current_price,
            "historical_volatility": historical_vol,
            "implied_volatility": implied_vol,
            "days_to_expiration": avg_dte,
            "dividend_yield": div_yield,
            "risk_free_rate": risk_free_rate
        }

@dataclass
class OptionParameters:
    """
    Option pricing parameters with SPY data integration
    """
    spot_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    maturity: float  # In years
    dividend_yield: float = 0.0
    option_type: str = "put"
    model_seed: int = 42
    
    # Advanced configuration parameters
    num_paths: int = 50000
    num_timesteps: int = 252  # One trading day per timestep
    confidence_level: float = 0.95
    regression_degree: int = 3
    regularization_strength: float = 1.0
    
    @classmethod
    def from_spy_data(cls, market_params: Dict[str, float], strike_price: Optional[float] = None):
        """
        Create option parameters directly from SPY market data
        
        Parameters:
        - market_params: Dictionary with market parameters
        - strike_price: Optional strike price (defaults to ATM)
        """
        if strike_price is None:
            strike_price = market_params["spot_price"]
            
        return cls(
            spot_price=market_params["spot_price"],
            strike_price=strike_price,
            risk_free_rate=market_params["risk_free_rate"],
            volatility=market_params["implied_volatility"],
            maturity=market_params["days_to_expiration"] / 252,  # Convert to years
            dividend_yield=market_params["dividend_yield"]
        )

class LeastSquaresMonteCarloPricer:
    """
    Advanced Least Squares Monte Carlo pricer for American options
    """
    
    def __init__(self, params: OptionParameters):
        """
        Initialize with option parameters
        
        Parameters:
        - params: Option pricing parameters
        """
        self.params = params
        self.logger = logger
        self.model_id = hashlib.md5(str(params.__dict__).encode()).hexdigest()[:8]
        self.logger.info(f"Initializing LSMC Pricer (ID: {self.model_id})")
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _generate_stock_paths(
        S0: float, 
        r: float, 
        sigma: float, 
        T: float, 
        div: float,
        n_steps: int,
        n_paths: int,
        seed: int
    ) -> np.ndarray:
        """
        JIT-accelerated stock path generation
        
        Parameters:
        - S0: Initial stock price
        - r: Risk-free rate
        - sigma: Volatility
        - T: Time to maturity (years)
        - div: Dividend yield
        - n_steps: Number of time steps
        - n_paths: Number of simulation paths
        - seed: Random seed
        
        Returns:
        - Array of simulated stock paths
        """
        np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in prange(n_paths):
            for t in range(1, n_steps + 1):
                z = np.random.standard_normal()
                paths[i, t] = paths[i, t-1] * np.exp((r - div - 0.5 * sigma**2) * dt + 
                                                     sigma * np.sqrt(dt) * z)
        
        return paths
    
    def generate_stock_paths(self) -> np.ndarray:
        """
        Generate stock price paths
        
        Returns:
        - Array of simulated stock paths
        """
        return self._generate_stock_paths(
            self.params.spot_price,
            self.params.risk_free_rate,
            self.params.volatility,
            self.params.maturity,
            self.params.dividend_yield,
            self.params.num_timesteps,
            self.params.num_paths,
            self.params.model_seed
        )
    
    def price_option(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Price American option using Least Squares Monte Carlo
        
        Returns:
        - Dictionary with pricing results
        """
        start_time = time.time()
        
        # Generate stock paths
        stock_paths = self.generate_stock_paths()
        
        # Calculate time step
        dt = self.params.maturity / self.params.num_timesteps
        
        # Set up discount factor
        discount_factor = np.exp(-self.params.risk_free_rate * dt)
        
        # Initialize payoff matrix
        if self.params.option_type == "put":
            payoffs = np.maximum(self.params.strike_price - stock_paths, 0)
        else:  # call option
            payoffs = np.maximum(stock_paths - self.params.strike_price, 0)
        
        # Initialize continuation values
        continuation_values = np.zeros_like(stock_paths)
        
        # Exercise decisions (1 = exercise, 0 = continue)
        exercise_decision = np.zeros_like(stock_paths)
        
        # Backward induction
        for t in range(self.params.num_timesteps - 1, 0, -1):
            # Discount future payoffs
            discounted_future = payoffs[:, t+1] * discount_factor
            
            # Find in-the-money paths
            itm_indices = payoffs[:, t] > 0
            
            if np.sum(itm_indices) > 0:
                # Extract in-the-money stock prices
                X = stock_paths[itm_indices, t].reshape(-1, 1)
                
                # Generate polynomial features
                poly = PolynomialFeatures(degree=self.params.regression_degree)
                X_poly = poly.fit_transform(X)
                
                # Perform regression
                model = Ridge(alpha=self.params.regularization_strength)
                model.fit(X_poly, discounted_future[itm_indices])
                
                # Predict continuation values
                predicted_continuation = model.predict(X_poly)
                
                # Determine exercise decision
                exercise = payoffs[itm_indices, t] > predicted_continuation
                
                # Update continuation values
                continuation_values[itm_indices, t] = predicted_continuation
                
                # Update exercise decision
                exercise_decision[itm_indices, t] = exercise
                
                # Update payoffs
                payoffs[itm_indices, t] = np.where(
                    exercise,
                    payoffs[itm_indices, t],
                    discounted_future[itm_indices]
                )
            else:
                # No in-the-money paths at this time step
                payoffs[:, t] = discounted_future
        
        # Calculate option price
        option_price = np.mean(payoffs[:, 1])
        option_std = np.std(payoffs[:, 1]) / np.sqrt(self.params.num_paths)
        
        # Calculate confidence interval
        ci_lower = option_price - stats.norm.ppf(0.975) * option_std
        ci_upper = option_price + stats.norm.ppf(0.975) * option_std
        
        # Performance metrics
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Option priced in {elapsed_time:.4f} seconds")
        self.logger.info(f"Option price: ${option_price:.4f} ± ${option_std:.4f}")
        
        return {
            "price": option_price,
            "std_error": option_std,
            "confidence_interval": (ci_lower, ci_upper),
            "paths": stock_paths,
            "payoffs": payoffs,
            "exercise_decision": exercise_decision,
            "computation_time": elapsed_time
        }
    
    def plot_results(self, results: Dict[str, Union[float, np.ndarray]]) -> None:
        """
        Plot pricing results for analysis
        
        Parameters:
        - results: Dictionary with pricing results
        """
        # Plot sample paths
        plt.figure(figsize=(12, 8))
        
        # Plot a subset of stock paths
        sample_paths = results["paths"][:100]
        for i in range(min(20, len(sample_paths))):
            plt.plot(sample_paths[i], alpha=0.3, lw=0.5)
        
        plt.plot(np.mean(sample_paths, axis=0), 'r--', lw=2, label='Average Path')
        plt.title(f"Sample Stock Price Paths - {self.params.option_type.capitalize()} Option")
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.axhline(y=self.params.strike_price, color='k', linestyle='-', alpha=0.3, 
                    label=f'Strike Price (${self.params.strike_price:.2f})')
        plt.legend()
        
        # Save plot
        plt.savefig(f"stock_paths_{self.model_id}.png")
        plt.close()
        
        # Plot exercise boundary
        plt.figure(figsize=(12, 8))
        
        # For each time step, find the average stock price where exercise occurs
        exercise_boundary = []
        time_steps = []
        
        for t in range(1, self.params.num_timesteps):
            exercise_indices = results["exercise_decision"][:, t] > 0
            if np.sum(exercise_indices) > 10:  # Only if we have enough data points
                avg_exercise_price = np.mean(results["paths"][exercise_indices, t])
                exercise_boundary.append(avg_exercise_price)
                time_steps.append(t)
        
        if exercise_boundary:
            plt.plot(time_steps, exercise_boundary, 'bo-', label='Exercise Boundary')
            plt.axhline(y=self.params.strike_price, color='r', linestyle='--', 
                        label=f'Strike Price (${self.params.strike_price:.2f})')
            plt.title(f"Exercise Boundary - {self.params.option_type.capitalize()} Option")
            plt.xlabel("Time Step")
            plt.ylabel("Stock Price")
            plt.legend()
            
            # Save plot
            plt.savefig(f"exercise_boundary_{self.model_id}.png")
        plt.close()

def main():
    """
    Main function to demonstrate SPY option pricing model
    """
    # Define paths
    optiondx_pwd = "/Users/nicolas/desktop/quantitative_finance/FEC/ml-american-options/optiondx/"
    
    # Initialize SPY data handler
    spy_handler = SPYDataHandler(optiondx_pwd)
    
    # Load January 2023 SPY data
    spy_jan_data = spy_handler.load_data("q1", "01")
    
    # Process data
    processed_data = spy_handler.process_data(spy_jan_data)
    
    # Extract market parameters
    market_params = spy_handler.extract_market_parameters(processed_data)
    
    # Create option parameters for ATM put
    atm_put_params = OptionParameters.from_spy_data(
        market_params,
        strike_price=market_params["spot_price"]
    )
    atm_put_params.option_type = "put"
    
    # Create option parameters for ATM call
    atm_call_params = OptionParameters.from_spy_data(
        market_params,
        strike_price=market_params["spot_price"]
    )
    atm_call_params.option_type = "call"
    
    # Price ATM put
    logger.info("Pricing ATM Put Option")
    put_pricer = LeastSquaresMonteCarloPricer(atm_put_params)
    put_results = put_pricer.price_option()
    put_pricer.plot_results(put_results)
    
    # Price ATM call
    logger.info("Pricing ATM Call Option")
    call_pricer = LeastSquaresMonteCarloPricer(atm_call_params)
    call_results = call_pricer.price_option()
    call_pricer.plot_results(call_results)
    
    # Compare with market prices (if available)
    atm_puts = processed_data["puts"][
        abs(processed_data["puts"]["[STRIKE]"] - market_params["spot_price"]) < 1
    ]
    
    atm_calls = processed_data["calls"][
        abs(processed_data["calls"]["[STRIKE]"] - market_params["spot_price"]) < 1
    ]
    
    # Print results
    print("\n" + "="*50)
    print("SPY OPTION PRICING RESULTS")
    print("="*50)
    
    print(f"\nMarket Parameters:")
    print(f"SPY Price: ${market_params['spot_price']:.2f}")
    print(f"Implied Volatility: {market_params['implied_volatility']*100:.2f}%")
    print(f"Historical Volatility: {market_params['historical_volatility']*100:.2f}%")
    print(f"Days to Expiration: {market_params['days_to_expiration']:.0f}")
    print(f"Risk-Free Rate: {market_params['risk_free_rate']*100:.2f}%")
    
    print("\nPut Option Results:")
    print(f"Model Price: ${put_results['price']:.4f}")
    print(f"95% Confidence Interval: (${put_results['confidence_interval'][0]:.4f}, "
          f"${put_results['confidence_interval'][1]:.4f})")
    
    if len(atm_puts) > 0:
        market_price = atm_puts["[P_LAST]"].iloc[-1]
        print(f"Market Price: ${market_price:.4f}")
        print(f"Model-Market Difference: ${put_results['price'] - market_price:.4f}")
    
    print("\nCall Option Results:")
    print(f"Model Price: ${call_results['price']:.4f}")
    print(f"95% Confidence Interval: (${call_results['confidence_interval'][0]:.4f}, "
          f"${call_results['confidence_interval'][1]:.4f})")
    
    if len(atm_calls) > 0:
        market_price = atm_calls["[C_LAST]"].iloc[-1]
        print(f"Market Price: ${market_price:.4f}")
        print(f"Model-Market Difference: ${call_results['price'] - market_price:.4f}")
    
    print("\nComputation Performance:")
    print(f"Put Option: {put_results['computation_time']:.2f} seconds")
    print(f"Call Option: {call_results['computation_time']:.2f} seconds")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Parameter': ['Spot Price', 'Strike Price', 'Risk-Free Rate', 'Implied Volatility',
                     'Days to Expiration', 'Put Price', 'Put Std Error', 'Call Price', 'Call Std Error',
                     'Computation Time (Put)', 'Computation Time (Call)'],
        'Value': [market_params['spot_price'], atm_put_params.strike_price, 
                 market_params['risk_free_rate'], market_params['implied_volatility'],
                 market_params['days_to_expiration'], put_results['price'], put_results['std_error'],
                 call_results['price'], call_results['std_error'], 
                 put_results['computation_time'], call_results['computation_time']]
    })
    
    results_df.to_csv(f"option_pricing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    logger.info("Results saved to CSV")

if __name__ == "__main__":
    main()
