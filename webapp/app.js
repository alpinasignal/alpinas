// Alpina Signal - Mini App JavaScript
// Professional Yellow/Black Design with TradingView Integration

// Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';
const FREE_ATTEMPTS_LIMIT = 2;

// Telegram Web App
const tg = window.Telegram?.WebApp || { expand: () => {}, ready: () => {}, initDataUnsafe: {} };
tg.expand();
tg.ready();

// State
let attemptsUsed = 0;
let currentTimeframe = '5';
let selectedCoin = 'BTCUSDT';
let userId = tg.initDataUnsafe?.user?.id || 123456789;
let tvWidget = null;
let tvChart = null;  // Store chart reference

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadAttempts();
    initializeTradingView();
    loadUserInfo();
});

// Initialize app
function initializeApp() {
    updateAttemptsDisplay();
}

// Load user info from Telegram
function loadUserInfo() {
    const telegramUser = tg.initDataUnsafe?.user;

    const usernameEl = document.getElementById('username');
    const userIdEl = document.getElementById('userId');

    if (telegramUser && telegramUser.id) {
        // Update username
        if (telegramUser.username) {
            usernameEl.textContent = `@${telegramUser.username}`;
        } else if (telegramUser.first_name) {
            usernameEl.textContent = telegramUser.first_name;
        } else {
            usernameEl.textContent = `User ${telegramUser.id}`;
        }

        // Update user ID
        userIdEl.textContent = telegramUser.id;
        userId = telegramUser.id;
    } else {
        // Fallback for testing outside Telegram
        usernameEl.textContent = 'Demo User';
        userIdEl.textContent = '123456789';
        userId = 123456789;
    }
}

// Initialize TradingView Widget
function initializeTradingView() {
    if (typeof TradingView === 'undefined') {
        console.error('TradingView library not loaded');
        return;
    }

    tvWidget = new TradingView.widget({
        container_id: 'tradingview_chart',
        width: '100%',
        height: 300,
        symbol: `BINANCE:${selectedCoin}`,
        interval: currentTimeframe,
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#16181d',
        enable_publishing: false,
        hide_top_toolbar: false,
        hide_legend: true,
        save_image: false,
        backgroundColor: '#16181d',
        gridColor: '#2a2d35',
        studies: [],
        disabled_features: [
            'header_widget',
            'timeframes_toolbar',
            'volume_force_overlay',
            'create_volume_indicator_by_default'
        ],
        enabled_features: [],
        overrides: {
            'mainSeriesProperties.candleStyle.upColor': '#00e676',
            'mainSeriesProperties.candleStyle.downColor': '#ff5252',
            'mainSeriesProperties.candleStyle.borderUpColor': '#00e676',
            'mainSeriesProperties.candleStyle.borderDownColor': '#ff5252',
            'mainSeriesProperties.candleStyle.wickUpColor': '#00e676',
            'mainSeriesProperties.candleStyle.wickDownColor': '#ff5252'
        }
    });

    // Store chart reference when widget is ready
    tvWidget.onChartReady(() => {
        tvChart = tvWidget.activeChart();
        console.log('TradingView chart is ready');
    });

    // Update price display immediately and then every 5 seconds
    updatePriceDisplay();
    setInterval(updatePriceDisplay, 5000);
}

// Update TradingView chart
function updateTradingViewChart() {
    if (!tvChart) {
        console.log('Chart not ready yet, waiting...');
        // Try again after a short delay
        setTimeout(() => {
            if (tvWidget) {
                tvChart = tvWidget.activeChart();
                if (tvChart) {
                    updateTradingViewChart();
                }
            }
        }, 500);
        return;
    }

    try {
        // Update symbol
        tvChart.setSymbol(`BINANCE:${selectedCoin}`, () => {
            console.log(`Chart symbol updated to ${selectedCoin}`);
        });

        // Update resolution (timeframe)
        tvChart.setResolution(currentTimeframe, () => {
            console.log(`Chart timeframe updated to ${currentTimeframe}m`);
        });
    } catch (error) {
        console.error('Error updating TradingView chart:', error);
    }
}

// Update price display
async function updatePriceDisplay() {
    try {
        const response = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${selectedCoin}`);
        const data = await response.json();

        if (data.price) {
            const price = parseFloat(data.price);

            // Format price based on value
            let formattedPrice;
            if (price >= 1000) {
                formattedPrice = price.toFixed(2);
            } else if (price >= 1) {
                formattedPrice = price.toFixed(4);
            } else {
                formattedPrice = price.toFixed(6);
            }

            document.getElementById('chartPrice').textContent = `$${formattedPrice}`;
        }
    } catch (error) {
        console.error('Error fetching price:', error);
        document.getElementById('chartPrice').textContent = 'Loading...';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Timeframe buttons
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimeframe = btn.dataset.time;
            updateTradingViewChart();
        });
    });

    // Coin select
    document.getElementById('coinSelect').addEventListener('change', async (e) => {
        selectedCoin = e.target.value;
        updateChartInfo();
        updateTradingViewChart();
        // Force immediate price update
        await updatePriceDisplay();
    });

    // Get Signal button
    document.getElementById('signalBtn').addEventListener('click', handleGetSignal);
}

// Open support (Telegram @alpinasignal)
function openSupport() {
    window.open('https://t.me/alpinasignal', '_blank');

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('light');
    }
}

// Load attempts from localStorage
function loadAttempts() {
    const stored = localStorage.getItem('alpina_attempts');
    if (stored) {
        const data = JSON.parse(stored);
        const today = new Date().toDateString();

        if (data.date === today) {
            attemptsUsed = data.count;
        } else {
            // Reset if new day
            attemptsUsed = 0;
            saveAttempts();
        }
    }
    updateAttemptsDisplay();
}

// Save attempts to localStorage
function saveAttempts() {
    const data = {
        count: attemptsUsed,
        date: new Date().toDateString()
    };
    localStorage.setItem('alpina_attempts', JSON.stringify(data));
}

// Update attempts display
function updateAttemptsDisplay() {
    const attemptsLeft = FREE_ATTEMPTS_LIMIT - attemptsUsed;
    document.getElementById('attemptsCount').textContent = `${attemptsLeft} / ${FREE_ATTEMPTS_LIMIT}`;
}

// Handle Get Signal button
async function handleGetSignal() {
    // Check if user has attempts left
    if (attemptsUsed >= FREE_ATTEMPTS_LIMIT) {
        showSubscriptionModal();
        return;
    }

    // Show loading
    showLoading();

    // Simulate signal generation (mock data for now)
    setTimeout(() => {
        hideLoading();

        // Mock signal data
        const signals = [
            { signal: 'LONG', confidence: 0.73 },
            { signal: 'SHORT', confidence: 0.68 },
            { signal: 'NO TRADE', confidence: 0.45 }
        ];

        const randomSignal = signals[Math.floor(Math.random() * signals.length)];

        // Increment attempts
        attemptsUsed++;
        saveAttempts();
        updateAttemptsDisplay();

        // Show result
        showSignalResult(randomSignal);

        // Haptic feedback
        if (tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('success');
        }

        // If this was the last free attempt, show modal after a delay
        if (attemptsUsed >= FREE_ATTEMPTS_LIMIT) {
            setTimeout(() => {
                showSubscriptionModal();
            }, 3000);
        }
    }, 2000);
}

// Show signal result
function showSignalResult(data) {
    const resultBox = document.getElementById('resultBox');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');

    resultBox.classList.remove('success', 'error');

    if (data.signal === 'LONG') {
        resultBox.classList.add('success');
        resultIcon.textContent = '📈';
        resultText.textContent = `LONG Signal (${(data.confidence * 100).toFixed(1)}%)`;
    } else if (data.signal === 'SHORT') {
        resultBox.classList.add('error');
        resultIcon.textContent = '📉';
        resultText.textContent = `SHORT Signal (${(data.confidence * 100).toFixed(1)}%)`;
    } else {
        resultIcon.textContent = '⚠️';
        resultText.textContent = 'NO TRADE - Wait for better setup';
    }

    resultBox.style.display = 'block';

    // Update price
    updatePriceDisplay();
}

// Show error result
function showErrorResult() {
    const resultBox = document.getElementById('resultBox');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');

    resultBox.classList.remove('success');
    resultBox.classList.add('error');
    resultIcon.textContent = '⚠️';
    resultText.textContent = 'Server error. Please try again.';
    resultBox.style.display = 'block';
}

// Show subscription modal
function showSubscriptionModal() {
    const modal = document.getElementById('subscriptionModal');
    modal.classList.add('show');

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('warning');
    }
}

// Payment state
let selectedPlanName = '';
let selectedPlanAmount = 0;

// Wallet address (USDT TRC20)
const WALLET_ADDRESS = 'TECGFKQd1SuJdVihGnegeVGfEXKnpCWieY';

// Close modal
function closeModal() {
    const modal = document.getElementById('subscriptionModal');
    modal.classList.remove('show');

    // Reset to plan selection step
    document.getElementById('paymentStep').style.display = 'none';
    document.getElementById('planStep').style.display = 'block';
}

// Select a subscription plan
function selectPlan(plan, amount) {
    selectedPlanName = plan;
    selectedPlanAmount = amount;

    // Hide plan selection, show payment step
    document.getElementById('planStep').style.display = 'none';
    document.getElementById('paymentStep').style.display = 'block';

    // Update payment amount
    document.getElementById('paymentAmount').textContent = `$${amount.toFixed(2)}`;

    // Generate QR code for the wallet address
    const qrCodeUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${WALLET_ADDRESS}`;
    document.getElementById('qrCode').src = qrCodeUrl;

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('light');
    }
}

// Copy wallet address to clipboard
function copyAddress() {
    const address = WALLET_ADDRESS;

    // Try modern clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(address).then(() => {
            alert('✅ Address copied to clipboard!');
        }).catch(() => {
            // Fallback
            fallbackCopyAddress(address);
        });
    } else {
        fallbackCopyAddress(address);
    }

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }
}

// Fallback copy method
function fallbackCopyAddress(address) {
    const textArea = document.createElement('textarea');
    textArea.value = address;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    try {
        document.execCommand('copy');
        alert('✅ Address copied to clipboard!');
    } catch (err) {
        alert('❌ Failed to copy. Please copy manually:\n' + address);
    }
    document.body.removeChild(textArea);
}

// Check payment status
async function checkPayment() {
    // Show loading
    showLoading();

    try {
        // Call backend to verify payment
        const response = await fetch(`${API_BASE_URL}/verify-payment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                plan: selectedPlanName,
                amount: selectedPlanAmount,
                wallet_address: WALLET_ADDRESS
            })
        });

        const data = await response.json();

        hideLoading();

        if (data.success && data.payment_verified) {
            // Payment verified - activate subscription
            alert('🎉 Payment verified!\n\nYour subscription is now active.\nThank you!');

            // Reset attempts (give unlimited)
            attemptsUsed = 0;
            saveAttempts();
            updateAttemptsDisplay();

            // Close modal
            closeModal();

            // Haptic feedback
            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('success');
            }
        } else {
            // Payment not found yet
            alert('⏳ Payment not detected yet\n\nPlease wait 1-5 minutes after sending USDT.\n\nIf you just sent it, please try again in a few moments.');

            // Haptic feedback
            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('warning');
            }
        }
    } catch (error) {
        hideLoading();
        console.error('Payment verification error:', error);

        // For demo purposes, activate subscription anyway
        alert('⚠️ Payment verification service unavailable\n\nPlease contact support at @alpinasignal');

        // Haptic feedback
        if (tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('error');
        }
    }
}

// Back to plan selection
function backToPlans() {
    document.getElementById('paymentStep').style.display = 'none';
    document.getElementById('planStep').style.display = 'block';

    selectedPlanName = '';
    selectedPlanAmount = 0;
}

// Legacy subscribe function (deprecated)
function subscribe(plan) {
    selectPlan(plan, 24.99);  // Default to Pro plan amount
}

// Update chart info
function updateChartInfo() {
    const coinName = selectedCoin.replace('USDT', '');
    const names = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'SOL': 'Solana',
        'BNB': 'Binance Coin',
        'XRP': 'Ripple',
        'ADA': 'Cardano',
        'AVAX': 'Avalanche',
        'LINK': 'Chainlink',
        'DOT': 'Polkadot',
        'MATIC': 'Polygon',
        'LTC': 'Litecoin',
        'OP': 'Optimism',
        'ARB': 'Arbitrum',
        'DOGE': 'Dogecoin',
        'TRX': 'Tron'
    };

    const fullName = names[coinName] || coinName;
    document.getElementById('chartTitle').textContent = `🪙 ${fullName} / TetherUS`;
}

// Show loading
function showLoading() {
    document.getElementById('loadingOverlay').classList.add('show');
}

// Hide loading
function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

// Tab navigation functions
function showSignalsTab() {
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Show signals content, hide about
    document.getElementById('signalsContent').style.display = 'block';
    document.getElementById('aboutContent').style.display = 'none';
}

function showAboutTab() {
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Show about content, hide signals
    document.getElementById('signalsContent').style.display = 'none';
    document.getElementById('aboutContent').style.display = 'block';
}

// Expose functions globally for onclick handlers
window.openSupport = openSupport;
window.closeModal = closeModal;
window.subscribe = subscribe;
window.selectPlan = selectPlan;
window.copyAddress = copyAddress;
window.checkPayment = checkPayment;
window.backToPlans = backToPlans;
window.showSignalsTab = showSignalsTab;
window.showAboutTab = showAboutTab;
window.showSubscriptionModal = showSubscriptionModal;
