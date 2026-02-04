"""
Subscription and Payment System
Manages user subscriptions and access control
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, text
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
        self.database_url = self._fix_supabase_url(database_url)
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def _fix_supabase_url(self, url: str) -> str:
        """
        Convert Supabase direct connection to Session Pooler connection.
        This fixes IPv6 connectivity issues on Railway and other platforms.
        """
        if not url or 'sqlite' in url.lower():
            return url

        # If already using pooler, return as-is
        if 'pooler.supabase.com' in url:
            logger.info("Using Supabase Pooler connection")
            return url

        # Check if it's a Supabase URL (port 5432)
        if '.supabase.co' in url and ':5432' in url:
            try:
                import re
                # Extract project ref from db.PROJECT_REF.supabase.co
                match = re.search(r'db\.([a-z0-9]+)\.supabase\.co', url)
                if match:
                    project_ref = match.group(1)
                    logger.info(f"Detected Supabase project: {project_ref}")

                    # Extract password (between : and @)
                    pwd_match = re.search(r'postgres:([^@]+)@', url)
                    if pwd_match:
                        password = pwd_match.group(1)

                        # Store project info for multi-region retry
                        self._supabase_project = project_ref
                        self._supabase_password = password

                        # Default to EU Central (most Supabase projects)
                        region = "aws-0-eu-central-1"
                        pooler_url = f"postgresql://postgres.{project_ref}:{password}@{region}.pooler.supabase.com:6543/postgres?sslmode=require"

                        logger.info(f"Converted to Supabase Pooler URL (region: {region})")
                        return pooler_url

            except Exception as e:
                logger.warning(f"Could not convert Supabase URL to pooler: {e}")

        return url

    def _try_connect_with_regions(self):
        """Try connecting with different Supabase regions"""
        if not hasattr(self, '_supabase_project'):
            return False

        regions = [
            "aws-0-eu-central-1",   # Europe (Frankfurt)
            "aws-0-us-east-1",      # US East
            "aws-0-us-west-1",      # US West
            "aws-0-ap-southeast-1", # Asia
        ]

        for region in regions:
            try:
                test_url = f"postgresql://postgres.{self._supabase_project}:{self._supabase_password}@{region}.pooler.supabase.com:6543/postgres?sslmode=require"
                test_engine = create_engine(test_url, connect_args={'connect_timeout': 5})

                # Try to connect
                with test_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    logger.info(f"Successfully connected to Supabase via {region}")
                    self.database_url = test_url
                    return True

            except Exception as e:
                logger.debug(f"Region {region} failed: {e}")
                continue

        return False

    def _initialize(self):
        """Lazy initialization of database connection"""
        if self._initialized:
            return

        try:
            # Different connect_args for different databases
            if 'sqlite' in self.database_url.lower():
                connect_args = {}
                engine_kwargs = {
                    'connect_args': connect_args,
                    'pool_pre_ping': True
                }
            else:
                # PostgreSQL (Supabase)
                connect_args = {
                    'connect_timeout': 10,
                    'sslmode': 'require'
                }
                engine_kwargs = {
                    'connect_args': connect_args,
                    'pool_pre_ping': True,
                    'pool_size': 3,
                    'max_overflow': 5,
                    'pool_timeout': 15
                }

            # First attempt with current URL
            try:
                self.engine = create_engine(self.database_url, **engine_kwargs)
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")
            except Exception as conn_err:
                logger.warning(f"First connection attempt failed: {conn_err}")

                # Try different Supabase regions
                if hasattr(self, '_supabase_project') and self._try_connect_with_regions():
                    self.engine = create_engine(self.database_url, **engine_kwargs)
                else:
                    raise conn_err

            # Create tables
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)

            # Verify schema
            session = self.SessionLocal()
            try:
                session.query(User).limit(1).all()
                session.close()
                logger.info("Database schema verified OK")
            except Exception as schema_error:
                session.close()
                logger.warning(f"Schema mismatch: {schema_error}")
                logger.info("Recreating tables...")
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                logger.info("Tables recreated successfully")

            self._initialized = True
            logger.info("✓ Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            logger.warning("Running in fallback mode (SQLite)")
            # Fallback to local SQLite
            try:
                self.database_url = "sqlite:///./alpina_signal.db"
                self.engine = create_engine(self.database_url, connect_args={})
                Base.metadata.create_all(self.engine)
                self.SessionLocal = sessionmaker(bind=self.engine)
                self._initialized = True
                logger.info("✓ Fallback to SQLite successful")
            except Exception as sqlite_err:
                logger.error(f"SQLite fallback also failed: {sqlite_err}")
                self._initialized = False

    def get_session(self) -> Optional[Session]:
        """Get database session"""
        if not self._initialized:
            self._initialize()

        if self._initialized and self.SessionLocal:
            return self.SessionLocal()
        return None


# ========================
# SUBSCRIPTION MANAGER
# ========================

class SubscriptionManager:
    """Manages user subscriptions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def is_admin(self, telegram_id: int) -> bool:
        """Check if user is admin"""
        return telegram_id in config.ADMIN_IDS

    def create_user(
        self,
        telegram_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None
    ) -> Optional[User]:
        """Create new user with free subscription"""
        session = self.db.get_session()

        if not session:
            logger.warning(f"No database session, skipping user creation for {telegram_id}")
            return None

        try:
            # Check if user exists
            user = session.query(User).filter(User.telegram_id == telegram_id).first()

            if user:
                # Update last active
                user.last_active = datetime.utcnow()
                if username:
                    user.username = username
                if first_name:
                    user.first_name = first_name
                session.commit()
                # Re-query to get a fresh instance
                user = session.query(User).filter(User.telegram_id == telegram_id).first()
                return user

            # Create new user
            user = User(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name
            )
            session.add(user)
            session.flush()  # Get the ID assigned

            # Create free subscription
            self._create_free_subscription(session, user.id)

            session.commit()

            # Re-query to get a clean instance
            user = session.query(User).filter(User.telegram_id == telegram_id).first()

            logger.info(f"Created new user: {telegram_id}")
            return user

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            if session:
                session.rollback()
            return None
        finally:
            if session:
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

        if not session:
            return None

        try:
            subscription = session.query(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.is_active == True
            ).first()

            return subscription

        except Exception as e:
            logger.error(f"Error getting subscription: {e}")
            return None
        finally:
            if session:
                session.close()

    def can_access_symbol(
        self,
        telegram_id: int,
        user_id: int,
        symbol: str
    ) -> bool:
        """
        Check if user can access a symbol based on subscription

        Args:
            telegram_id: Telegram user ID (for admin check)
            user_id: Database user ID
            symbol: Trading pair

        Returns:
            True if user has access
        """
        # Check if admin - unlimited access
        if self.is_admin(telegram_id):
            return True

        subscription = self.get_subscription(user_id)

        if not subscription:
            # If no database, allow limited access
            return True

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

        if not session:
            return []

        try:
            history = session.query(PredictionHistory).filter(
                PredictionHistory.user_id == user_id
            ).order_by(PredictionHistory.created_at.desc()).limit(limit).all()

            return history

        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []
        finally:
            if session:
                session.close()

    # ========================
    # ADMIN FUNCTIONS
    # ========================

    def get_user_stats(self, telegram_id: int) -> Optional[dict]:
        """
        Get user statistics (admin only)

        Returns:
            Dictionary with user stats or None
        """
        if not self.is_admin(telegram_id):
            logger.warning(f"Non-admin {telegram_id} tried to access user stats")
            return None

        session = self.db.get_session()

        if not session:
            return {"error": "Database not available"}

        try:
            # Total users
            total_users = session.query(User).count()

            # Active users (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            active_users = session.query(User).filter(
                User.last_active >= week_ago
            ).count()

            # Users by subscription tier
            tier_counts = {}
            for tier in ["free", "starter", "basic", "pro", "premium"]:
                count = session.query(Subscription).filter(
                    Subscription.tier == tier,
                    Subscription.is_active == True
                ).count()
                tier_counts[tier] = count

            # Total predictions
            total_predictions = session.query(PredictionHistory).count()

            # Recent users (last 10)
            recent_users = session.query(User).order_by(
                User.created_at.desc()
            ).limit(10).all()

            recent_list = []
            for user in recent_users:
                subscription = session.query(Subscription).filter(
                    Subscription.user_id == user.id,
                    Subscription.is_active == True
                ).first()

                recent_list.append({
                    "telegram_id": user.telegram_id,
                    "username": user.username or "No username",
                    "first_name": user.first_name or "No name",
                    "created_at": user.created_at.strftime("%Y-%m-%d %H:%M"),
                    "tier": subscription.tier if subscription else "none"
                })

            return {
                "total_users": total_users,
                "active_users_7d": active_users,
                "tier_counts": tier_counts,
                "total_predictions": total_predictions,
                "recent_users": recent_list
            }

        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {"error": str(e)}
        finally:
            if session:
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
