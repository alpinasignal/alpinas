"""
Subscription and Payment System
Manages user subscriptions and access control
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from typing import Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

# Database setup
Base = declarative_base()


# ========================
# DATABASE MODELS
# ========================

class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)


class Subscription(Base):
    """Subscription model"""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)  # references User.id
    tier = Column(String, nullable=False)  # free, basic, pro, premium
    selected_coins = Column(JSON, nullable=True)  # List of selected coins
    predictions_used = Column(Integer, default=0)
    predictions_limit = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


class Payment(Base):
    """Payment history"""
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    tier = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="USDT")
    transaction_hash = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class PredictionHistory(Base):
    """Track prediction requests"""
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    signal = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ========================
# DATABASE MANAGER
# ========================

class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, database_url: str = config.DATABASE_URL):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()


# ========================
# SUBSCRIPTION MANAGER
# ========================

class SubscriptionManager:
    """Manages user subscriptions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_user(
        self,
        telegram_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None
    ) -> User:
        """Create new user with free subscription"""
        session = self.db.get_session()

        try:
            # Check if user exists
            user = session.query(User).filter(User.telegram_id == telegram_id).first()

            if user:
                # Update last active
                user.last_active = datetime.utcnow()
                session.commit()
                return user

            # Create new user
            user = User(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name
            )
            session.add(user)
            session.commit()
            session.refresh(user)

            # Create free subscription
            self._create_free_subscription(session, user.id)

            logger.info(f"Created new user: {telegram_id}")
            return user

        finally:
            session.close()

    def _create_free_subscription(self, session: Session, user_id: int):
        """Create free tier subscription"""
        free_tier = config.SUBSCRIPTION_TIERS["free"]

        subscription = Subscription(
            user_id=user_id,
            tier="free",
            predictions_used=0,
            predictions_limit=free_tier["predictions"],
            is_active=True
        )

        session.add(subscription)
        session.commit()

    def get_subscription(self, user_id: int) -> Optional[Subscription]:
        """Get active subscription for user"""
        session = self.db.get_session()

        try:
            subscription = session.query(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.is_active == True
            ).first()

            return subscription

        finally:
            session.close()

    def can_access_symbol(
        self,
        user_id: int,
        symbol: str
    ) -> bool:
        """
        Check if user can access a symbol based on subscription

        Args:
            user_id: User ID
            symbol: Trading pair

        Returns:
            True if user has access
        """
        subscription = self.get_subscription(user_id)

        if not subscription:
            return False

        # Check if subscription expired
        if subscription.expires_at and subscription.expires_at < datetime.utcnow():
            return False

        # Check if reached prediction limit
        if subscription.predictions_used >= subscription.predictions_limit:
            return False

        # Free tier: user must have selected this coin
        if subscription.tier == "free":
            selected = subscription.selected_coins or []
            return symbol in selected

        # Paid tier: check selected coins
        if subscription.selected_coins:
            return symbol in subscription.selected_coins

        return False

    def record_prediction(
        self,
        user_id: int,
        symbol: str,
        timeframe: str,
        signal: str,
        confidence: float,
        price: float
    ):
        """Record a prediction request"""
        session = self.db.get_session()

        try:
            # Increment predictions used
            subscription = session.query(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.is_active == True
            ).first()

            if subscription:
                subscription.predictions_used += 1
                session.commit()

            # Record in history
            history = PredictionHistory(
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                signal=signal,
                confidence=confidence,
                price=price
            )

            session.add(history)
            session.commit()

        finally:
            session.close()

    def upgrade_subscription(
        self,
        user_id: int,
        tier: str,
        selected_coins: List[str],
        duration_days: int = 30
    ) -> Subscription:
        """
        Upgrade user subscription

        Args:
            user_id: User ID
            tier: New tier
            selected_coins: List of selected coins
            duration_days: Subscription duration

        Returns:
            New subscription
        """
        session = self.db.get_session()

        try:
            # Deactivate old subscription
            old_subscription = session.query(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.is_active == True
            ).first()

            if old_subscription:
                old_subscription.is_active = False
                session.commit()

            # Create new subscription
            tier_config = config.SUBSCRIPTION_TIERS[tier]

            new_subscription = Subscription(
                user_id=user_id,
                tier=tier,
                selected_coins=selected_coins,
                predictions_used=0,
                predictions_limit=999999,  # Unlimited for paid tiers
                expires_at=datetime.utcnow() + timedelta(days=duration_days),
                is_active=True
            )

            session.add(new_subscription)
            session.commit()
            session.refresh(new_subscription)

            logger.info(f"Upgraded user {user_id} to {tier}")
            return new_subscription

        finally:
            session.close()

    def create_payment(
        self,
        user_id: int,
        tier: str,
        amount: float
    ) -> Payment:
        """Create payment record"""
        session = self.db.get_session()

        try:
            payment = Payment(
                user_id=user_id,
                tier=tier,
                amount=amount,
                status="pending"
            )

            session.add(payment)
            session.commit()
            session.refresh(payment)

            return payment

        finally:
            session.close()

    def complete_payment(
        self,
        payment_id: int,
        transaction_hash: str
    ):
        """Mark payment as completed"""
        session = self.db.get_session()

        try:
            payment = session.query(Payment).filter(Payment.id == payment_id).first()

            if payment:
                payment.status = "completed"
                payment.transaction_hash = transaction_hash
                payment.completed_at = datetime.utcnow()
                session.commit()

                logger.info(f"Payment {payment_id} completed")

        finally:
            session.close()

    def get_prediction_history(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[PredictionHistory]:
        """Get user's prediction history"""
        session = self.db.get_session()

        try:
            history = session.query(PredictionHistory).filter(
                PredictionHistory.user_id == user_id
            ).order_by(PredictionHistory.created_at.desc()).limit(limit).all()

            return history

        finally:
            session.close()


if __name__ == "__main__":
    # Test database setup
    db_manager = DatabaseManager()
    sub_manager = SubscriptionManager(db_manager)

    # Create test user
    user = sub_manager.create_user(
        telegram_id=123456789,
        username="testuser",
        first_name="Test"
    )

    print(f"Created user: {user.id}")

    # Get subscription
    subscription = sub_manager.get_subscription(user.id)
    print(f"Subscription tier: {subscription.tier}")
    print(f"Predictions limit: {subscription.predictions_limit}")

    # Test access
    can_access = sub_manager.can_access_symbol(user.id, "BTCUSDT")
    print(f"Can access BTCUSDT: {can_access}")
