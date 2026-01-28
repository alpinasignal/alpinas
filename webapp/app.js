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
let signalHistory = [];  // Store signal history
let currentPrice = 0;  // Current price for SL/TP calculation

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
    loadSignalHistory();
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
    console.log(`Attempting to update chart: ${selectedCoin}, timeframe: ${currentTimeframe}`);

    // Recreate widget with new symbol
    if (tvWidget) {
        try {
            // Remove old widget
            tvWidget.remove();
        } catch (e) {
            console.log('Error removing widget:', e);
        }
    }

    // Wait a bit before creating new widget
    setTimeout(() => {
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
            console.log(`TradingView chart updated to ${selectedCoin}`);
        });
    }, 100);
}

// Update price display
async function updatePriceDisplay() {
    try {
        const response = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${selectedCoin}`);
        const data = await response.json();

        if (data.price) {
            const price = parseFloat(data.price);
            currentPrice = price;  // Store for SL/TP calculation

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

    // Update price first
    await updatePriceDisplay();

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

        // Calculate SL/TP based on signal type
        let stopLoss = 0;
        let takeProfit = 0;
        const entryPrice = currentPrice;

        if (randomSignal.signal === 'LONG') {
            // LONG: SL below, TP above
            stopLoss = entryPrice * 0.98;  // 2% below
            takeProfit = entryPrice * 1.04;  // 4% above (Risk:Reward 1:2)
        } else if (randomSignal.signal === 'SHORT') {
            // SHORT: SL above, TP below
            stopLoss = entryPrice * 1.02;  // 2% above
            takeProfit = entryPrice * 0.96;  // 4% below
        }

        // Add SL/TP to signal data
        const signalData = {
            ...randomSignal,
            entryPrice,
            stopLoss,
            takeProfit,
            coin: selectedCoin,
            timeframe: currentTimeframe + 'm',
            timestamp: new Date().toISOString(),
            timestampFormatted: new Date().toLocaleString()
        };

        // Save to history
        if (randomSignal.signal !== 'NO TRADE') {
            signalHistory.unshift(signalData);
            localStorage.setItem('alpina_signal_history', JSON.stringify(signalHistory));
        }

        // Increment attempts
        attemptsUsed++;
        saveAttempts();
        updateAttemptsDisplay();

        // Show result
        showSignalResult(signalData);

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
    const resultSubtext = document.getElementById('resultSubtext');
    const slTpBox = document.getElementById('slTpBox');

    resultBox.classList.remove('success', 'error');

    if (data.signal === 'LONG') {
        resultBox.classList.add('success');
        resultIcon.textContent = '📈';
        resultText.textContent = `LONG Signal (${(data.confidence * 100).toFixed(1)}%)`;
        resultSubtext.textContent = `Open LONG position on ${data.coin}`;

        // Show SL/TP
        slTpBox.style.display = 'block';
        document.getElementById('entryPrice').textContent = `$${formatPrice(data.entryPrice)}`;
        document.getElementById('stopLoss').textContent = `$${formatPrice(data.stopLoss)}`;
        document.getElementById('takeProfit').textContent = `$${formatPrice(data.takeProfit)}`;

    } else if (data.signal === 'SHORT') {
        resultBox.classList.add('error');
        resultIcon.textContent = '📉';
        resultText.textContent = `SHORT Signal (${(data.confidence * 100).toFixed(1)}%)`;
        resultSubtext.textContent = `Open SHORT position on ${data.coin}`;

        // Show SL/TP
        slTpBox.style.display = 'block';
        document.getElementById('entryPrice').textContent = `$${formatPrice(data.entryPrice)}`;
        document.getElementById('stopLoss').textContent = `$${formatPrice(data.stopLoss)}`;
        document.getElementById('takeProfit').textContent = `$${formatPrice(data.takeProfit)}`;

    } else {
        resultIcon.textContent = '⚠️';
        resultText.textContent = 'NO TRADE - Wait for better setup';
        resultSubtext.textContent = 'Market conditions are not favorable';
        slTpBox.style.display = 'none';
    }

    resultBox.style.display = 'block';

    // Update price
    updatePriceDisplay();
}

// Format price helper
function formatPrice(price) {
    if (price >= 1000) {
        return price.toFixed(2);
    } else if (price >= 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(6);
    }
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

// Load signal history from localStorage
function loadSignalHistory() {
    const stored = localStorage.getItem('alpina_signal_history');
    if (stored) {
        signalHistory = JSON.parse(stored);
    }
    updateHistoryDisplay();
}

// Update history display
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    const emptyHistory = document.getElementById('emptyHistory');
    const historyCount = document.getElementById('historyCount');

    if (signalHistory.length === 0) {
        emptyHistory.style.display = 'block';
        historyList.style.display = 'none';
        historyCount.textContent = '0 signals';
        return;
    }

    emptyHistory.style.display = 'none';
    historyList.style.display = 'flex';
    historyCount.textContent = `${signalHistory.length} signals`;

    // Clear existing items
    historyList.innerHTML = '';

    // Add history items
    signalHistory.forEach((signal, index) => {
        const item = document.createElement('div');
        item.className = 'chart-box';
        item.style.padding = '16px';
        item.style.marginBottom = index === signalHistory.length - 1 ? '0' : '0';

        const signalClass = signal.signal === 'LONG' ? 'success' : 'error';
        const signalIcon = signal.signal === 'LONG' ? '📈' : '📉';
        const signalColor = signal.signal === 'LONG' ? 'var(--success)' : 'var(--error)';

        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 16px; font-weight: 600; color: ${signalColor}; margin-bottom: 4px;">
                        ${signalIcon} ${signal.signal}
                    </div>
                    <div style="font-size: 13px; color: var(--text-muted);">
                        ${signal.coin} • ${signal.timeframe}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: var(--text-muted);">
                        ${signal.timestampFormatted}
                    </div>
                    <div style="font-size: 13px; color: var(--yellow); margin-top: 4px;">
                        ${(signal.confidence * 100).toFixed(1)}% confidence
                    </div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; padding-top: 12px; border-top: 1px solid var(--border); font-size: 12px;">
                <div>
                    <div style="color: var(--text-muted); margin-bottom: 4px;">Entry</div>
                    <div style="color: var(--yellow); font-weight: 600;">$${formatPrice(signal.entryPrice)}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); margin-bottom: 4px;">Stop Loss</div>
                    <div style="color: var(--error); font-weight: 600;">$${formatPrice(signal.stopLoss)}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); margin-bottom: 4px;">Take Profit</div>
                    <div style="color: var(--success); font-weight: 600;">$${formatPrice(signal.takeProfit)}</div>
                </div>
            </div>
        `;

        historyList.appendChild(item);
    });
}

// Tab navigation functions
function showSignalsTab() {
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Show signals content, hide others
    document.getElementById('signalsContent').style.display = 'block';
    document.getElementById('historyContent').style.display = 'none';
    document.getElementById('aboutContent').style.display = 'none';
}

function showHistoryTab() {
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Show history content, hide others
    document.getElementById('signalsContent').style.display = 'none';
    document.getElementById('historyContent').style.display = 'block';
    document.getElementById('aboutContent').style.display = 'none';

    // Refresh history display
    updateHistoryDisplay();
}

function showAboutTab() {
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Show about content, hide others
    document.getElementById('signalsContent').style.display = 'none';
    document.getElementById('historyContent').style.display = 'none';
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
window.showHistoryTab = showHistoryTab;
window.showAboutTab = showAboutTab;
window.showSubscriptionModal = showSubscriptionModal;
