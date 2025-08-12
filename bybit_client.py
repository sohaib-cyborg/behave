import ccxt.async_support as ccxt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import logging
import asyncio      
import os

class BybitClient:
    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit client using CCXT async.
        
        Args:
            testnet: Whether to use Bybit testnet (default: True)
        """
        self.exchange = ccxt.bybit({
            'apiKey': os.getenv("BYBIT_API_KEY"),
            'secret': os.getenv("BYBIT_API_SECRET"),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 'spot', 'future', or 'swap'
                'adjustForTimeDifference': True,
            },
        })
        
        # Set testnet if needed
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def close(self):
        if hasattr(self, 'exchange') and self.exchange is not None:
            await self.exchange.close()
            self.exchange = None
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        limit: int = 100,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Timestamp in milliseconds (optional)
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and datetime index
        """
        try:
            if not hasattr(self.exchange, 'fetch_ohlcv'):
                raise AttributeError("Exchange does not support fetch_ohlcv method")
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit, params={"category": "spot"})
            await self.exchange.close()

            if not ohlcv:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime and set as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    async def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict: Ticker information
        """
        try:
            symbol = self._format_symbol(symbol)
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            raise
    
    async def get_balance(self) -> Dict:
        """
        Get account balance.
        
        Returns:
            Dict: Account balance information
        """
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            raise
    
    async def create_order(
        self, 
        symbol: str, 
        order_type: str, 
        side: str, 
        amount: float, 
        price: Optional[float] = None, 
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            order_type: Order type ('market', 'limit', etc.)
            side: 'buy' or 'sell'
            amount: Amount to buy/sell
            price: Price (required for limit orders)
            params: Additional parameters
            
        Returns:
            Dict: Order information
        """
        try:
            symbol = self._format_symbol(symbol)
            return await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get open orders.
        
        Args:
            symbol: Optional trading pair symbol
            
        Returns:
            List[Dict]: List of open orders
        """
        try:
            if symbol:
                symbol = self._format_symbol(symbol)
                return await self.exchange.fetch_open_orders(symbol)
            return await self.exchange.fetch_open_orders()
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (required for some exchanges)
            
        Returns:
            Dict: Cancellation result
        """
        try:
            if symbol:
                symbol = self._format_symbol(symbol)
                return await self.exchange.cancel_order(order_id, symbol)
            return await self.exchange.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            raise
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol to match Bybit's format if needed."""
        # Convert common formats to Bybit's format (e.g., 'BTC/USDT' -> 'BTC/USDT' for spot)
        # No conversion needed for most cases with CCXT
        return symbol.replace('/', '')
    
    async def get_coin_price_volume(
         self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price (close) and volume for a coin from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            start_date: Start date in 'YYYY-MM-DD' format (UTC)
            end_date: End date in 'YYYY-MM-DD' format (UTC)
            interval: Timeframe (default: '1d', can be '1h', '1m', etc.)

        Returns:
            pd.DataFrame with columns ['date', 'price', 'volume']
        """
        try:
            # Convert date strings to timestamps in milliseconds
            since = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)

            all_data = []
            fetch_limit = 1000
            fetch_since = since

            # Fetch in chunks until we reach end_date
            while True:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=interval,
                    since=fetch_since,
                    limit=fetch_limit,
                    params={"category": "spot"}
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                last_ts = ohlcv[-1][0]
                if last_ts >= end_ts:
                    break

                # Move the "since" forward to avoid overlap
                fetch_since = last_ts + 1

                # Small delay to respect rate limits
                await asyncio.sleep(self.exchange.rateLimit / 1000)

            if not all_data:
                return None

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
            df.rename(columns={'close': 'price'}, inplace=True)
            return df[['date', 'price', 'volume']]

        except Exception as e:
            self.logger.error(f"Bybit fetch failed for {symbol}: {e}", exc_info=True)
            return None
    def load_markets(self):
        return self.exchange.load_markets()
    
    async def close(self):
        """Close the connection."""
        await self.exchange.close()
