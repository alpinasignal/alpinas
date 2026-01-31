"""
TRON TRC20 Payment Verification
Automatically verifies USDT TRC20 payments on TRON blockchain
"""

import requests
from typing import Optional, Dict
from datetime import datetime, timedelta
from loguru import logger


class TronPaymentVerifier:
    """
    Verifies USDT TRC20 payments on TRON blockchain
    Uses TronGrid API
    """

    # USDT TRC20 contract address on TRON
    USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

    # TronGrid API endpoint
    TRONGRID_API = "https://api.trongrid.io"

    def __init__(self, wallet_address: str):
        """
        Initialize verifier

        Args:
            wallet_address: Your receiving wallet address (TRC20)
        """
        self.wallet_address = wallet_address

    def verify_payment(
        self,
        expected_amount: float,
        timeframe_hours: int = 24
    ) -> Optional[Dict]:
        """
        Check if payment was received in the last N hours

        Args:
            expected_amount: Expected payment amount in USDT
            timeframe_hours: How many hours back to check

        Returns:
            Transaction details if found, None otherwise
        """
        try:
            # Get recent transactions to wallet
            url = f"{self.TRONGRID_API}/v1/accounts/{self.wallet_address}/transactions/trc20"
            params = {
                "limit": 50,  # Check last 50 transactions
                "contract_address": self.USDT_CONTRACT
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"TronGrid API error: {response.status_code}")
                return None

            data = response.json()

            if "data" not in data:
                logger.warning("No transaction data received from TronGrid")
                return None

            transactions = data["data"]

            # Current time
            now = datetime.utcnow()
            cutoff_time = now - timedelta(hours=timeframe_hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)  # TronGrid uses milliseconds

            # Check each transaction
            for tx in transactions:
                # Extract transaction details
                tx_timestamp = tx.get("block_timestamp", 0)
                tx_to = tx.get("to", "")
                tx_value = tx.get("value", "0")
                tx_hash = tx.get("transaction_id", "")

                # Convert value from smallest unit (6 decimals for USDT)
                tx_amount = float(tx_value) / 1_000_000  # USDT has 6 decimals

                # Check if transaction matches criteria
                is_to_us = tx_to.lower() == self.wallet_address.lower()
                is_recent = tx_timestamp >= cutoff_timestamp
                is_correct_amount = abs(tx_amount - expected_amount) < 0.01  # Allow 0.01 USDT tolerance

                logger.info(f"Checking TX {tx_hash[:10]}...: amount={tx_amount:.2f} USDT, to_us={is_to_us}, recent={is_recent}")

                if is_to_us and is_recent and is_correct_amount:
                    # Payment found!
                    logger.info(f"✅ Payment verified: {tx_amount:.2f} USDT - TX: {tx_hash}")

                    return {
                        "verified": True,
                        "amount": tx_amount,
                        "transaction_hash": tx_hash,
                        "timestamp": datetime.fromtimestamp(tx_timestamp / 1000).isoformat(),
                        "from_address": tx.get("from", "unknown")
                    }

            # No matching payment found
            logger.info(f"No matching payment found for {expected_amount} USDT")
            return None

        except Exception as e:
            logger.error(f"Error verifying payment: {e}")
            return None

    def get_wallet_balance(self) -> Optional[float]:
        """
        Get USDT balance of wallet

        Returns:
            Balance in USDT, or None on error
        """
        try:
            url = f"{self.TRONGRID_API}/v1/accounts/{self.wallet_address}"

            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.error(f"TronGrid API error: {response.status_code}")
                return None

            data = response.json()

            # Get TRC20 balances
            trc20_balances = data.get("data", [{}])[0].get("trc20", [])

            for token in trc20_balances:
                if token.get("token_id") == self.USDT_CONTRACT:
                    balance = float(token.get("balance", "0")) / 1_000_000
                    return balance

            return 0.0

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None


if __name__ == "__main__":
    # Test verification
    WALLET_ADDRESS = "TECGFKQd1SuJdVihGnegeVGfEXKnpCWieY"

    verifier = TronPaymentVerifier(WALLET_ADDRESS)

    # Check balance
    balance = verifier.get_wallet_balance()
    if balance is not None:
        print(f"💰 Wallet balance: {balance:.2f} USDT")

    # Check for payment
    result = verifier.verify_payment(
        expected_amount=14.99,
        timeframe_hours=24
    )

    if result:
        print(f"✅ Payment verified!")
        print(f"   Amount: {result['amount']:.2f} USDT")
        print(f"   TX: {result['transaction_hash']}")
        print(f"   Time: {result['timestamp']}")
    else:
        print("❌ No matching payment found")
