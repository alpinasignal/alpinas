// Alpina Signal - Mini App JavaScript
// Professional Yellow/Black Design with TradingView Integration

// Configuration
// Dynamic API URL - works on both localhost and Railway
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000/api/v1'
    : `${window.location.origin}/api/v1`;
const FREE_ATTEMPTS_LIMIT = 2;
const ADMIN_IDS = [7940666073, 5480212100];  // Admin Telegram IDs

// Telegram Web App
const tg = window.Telegram?.WebApp || { expand: () => {}, ready: () => {}, initDataUnsafe: {} };

// State
let attemptsUsed = 0;
let currentTimeframe = '15';
let selectedCoin = 'BTCUSDT';
let userId = 123456789;  // Will be set in loadUserInfo()
let tvWidget = null;
let tvChart = null;  // Store chart reference
let signalHistory = [];  // Store signal history
let currentPrice = 0;  // Current price for SL/TP calculation

// Map TradingView intervals to API timeframes
const TIMEFRAME_MAP = {
    '15': '15m',
    '60': '1h',
    '240': '4h'
};

// Wait for Telegram WebApp to be ready
console.log('Waiting for Telegram WebApp...');
tg.ready();
tg.expand();

// Initialize after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded');
    console.log('API Base URL:', API_BASE_URL);
    console.log('Telegram WebApp:', tg);
    console.log('Telegram initDataUnsafe:', tg.initDataUnsafe);

    // CRITICAL: Setup event listeners FIRST (before anything can crash)
    try { setupEventListeners(); } catch(e) { console.error('setupEventListeners error:', e); }
    try { initializeTradingView(); } catch(e) { console.error('initializeTradingView error:', e); }
    try { loadAttempts(); } catch(e) { console.error('loadAttempts error:', e); }
    try { loadSignalHistory(); } catch(e) { console.error('loadSignalHistory error:', e); }

    // Load user info LAST (admin check may modify DOM)
    loadUserInfo();

    // Apply admin UI if already detected
    applyAdminUI();
});

// Check if user is admin
function isAdmin() {
    return ADMIN_IDS.includes(userId);
}

// Apply admin UI (safe to call multiple times)
function applyAdminUI() {
    if (!isAdmin()) return;
    try {
        const banner = document.querySelector('.attempts-banner span');
        if (banner) banner.innerHTML = '👑 <strong>Admin Access - Unlimited</strong>';
        const adminBtn = document.getElementById('adminBtn');
        if (adminBtn) adminBtn.style.display = 'block';
        console.log('Admin UI applied');
    } catch(e) {
        console.error('applyAdminUI error:', e);
    }
}

// Load user info from Telegram
function loadUserInfo() {
    const usernameEl = document.getElementById('username');
    const userIdEl = document.getElementById('userId');

    console.log('Loading user info...');
    console.log('Telegram WebApp object:', tg);
    console.log('initDataUnsafe:', tg.initDataUnsafe);

    // Try to get user data immediately
    let telegramUser = tg.initDataUnsafe?.user;

    // If no user data yet, wait and retry (Telegram WebApp might still be loading)
    if (!telegramUser || !telegramUser.id) {
        console.log('User data not ready yet, retrying in 500ms...');
        setTimeout(() => {
            telegramUser = tg.initDataUnsafe?.user;
            console.log('Retry - Telegram User:', telegramUser);
            applyUserInfo(telegramUser, usernameEl, userIdEl);
        }, 500);
    } else {
        applyUserInfo(telegramUser, usernameEl, userIdEl);
    }
}

// Helper function to apply user info to DOM
function applyUserInfo(telegramUser, usernameEl, userIdEl) {
    if (telegramUser && telegramUser.id) {
        // Update username
        if (telegramUser.username) {
            usernameEl.textContent = `@${telegramUser.username}`;
            console.log('✅ Username set:', telegramUser.username);
        } else if (telegramUser.first_name) {
            usernameEl.textContent = telegramUser.first_name;
            console.log('✅ First name set:', telegramUser.first_name);
        } else {
            usernameEl.textContent = `User ${telegramUser.id}`;
            console.log('✅ Generic user name set');
        }

        // Update user ID
        userIdEl.textContent = telegramUser.id;
        userId = telegramUser.id;
        console.log('✅ User ID set:', userId);

        // Apply admin UI if this user is admin
        applyAdminUI();
    } else {
        // Fallback for testing outside Telegram
        console.warn('⚠️ No Telegram user data - using demo mode');
        usernameEl.textContent = 'Demo User';
        userIdEl.textContent = '123456789';
        userId = 123456789;
    }

    // Force display update
    usernameEl.style.display = 'block';
    userIdEl.style.display = 'inline';

    // Register user in database (for admin panel stats)
    registerUser(telegramUser);
}

// Register user in backend database
async function registerUser(telegramUser) {
    if (!telegramUser || !telegramUser.id) return;

    try {
        await fetch(`${API_BASE_URL}/register-user`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                telegram_id: telegramUser.id,
                username: telegramUser.username || null,
                first_name: telegramUser.first_name || null
            })
        });
        console.log('User registered in DB');
    } catch (e) {
        console.warn('Failed to register user:', e);
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
    // Timeframe buttons (only within signals content, not admin panel)
    document.querySelectorAll('#signalsContent .time-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#signalsContent .time-btn').forEach(b => b.classList.remove('active'));
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
    const el = document.getElementById('attemptsCount');
    if (!el) return; // Admin mode - attemptsCount element was replaced
    const attemptsLeft = FREE_ATTEMPTS_LIMIT - attemptsUsed;
    el.textContent = `${attemptsLeft} / ${FREE_ATTEMPTS_LIMIT}`;
}

// Handle Get Signal button
async function handleGetSignal() {
    // Skip limit check for admins
    if (!isAdmin()) {
        // Check if user has attempts left
        if (attemptsUsed >= FREE_ATTEMPTS_LIMIT) {
            showSubscriptionModal();
            return;
        }
    }

    // Update price first
    await updatePriceDisplay();

    // Show loading
    showLoading();

    // Map TradingView interval to API timeframe
    const apiTimeframe = TIMEFRAME_MAP[currentTimeframe] || '15m';
    const displayTimeframe = apiTimeframe;

    try {
        console.log('Calling AI API:', {
            url: `${API_BASE_URL}/predict`,
            symbol: selectedCoin,
            timeframe: apiTimeframe,
            user_id: userId
        });

        // Call REAL AI prediction API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'user-id': userId.toString()
            },
            body: JSON.stringify({
                symbol: selectedCoin,
                timeframe: apiTimeframe,
                user_id: userId
            })
        });

        console.log('API Response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error:', errorText);
            // Try to extract detail from JSON error
            let detail = errorText;
            try {
                const errJson = JSON.parse(errorText);
                detail = errJson.detail || errorText;
            } catch(e) {}
            throw new Error(`${response.status}::${detail}`);
        }

        const prediction = await response.json();
        console.log('AI Prediction received:', prediction);

        hideLoading();

        // Calculate SL/TP based on signal type and timeframe
        let stopLoss = 0;
        let takeProfit = 0;
        const entryPrice = prediction.price || currentPrice;

        // Dynamic SL/TP based on timeframe
        // Shorter timeframes = tighter SL/TP, longer = wider
        let slPercent, tpPercent;
        const tf = currentTimeframe || '60';
        if (tf === '15') {
            slPercent = 0.01;   // 1% SL
            tpPercent = 0.015;  // 1.5% TP (1:1.5 risk/reward)
        } else if (tf === '60') {
            slPercent = 0.015;  // 1.5% SL
            tpPercent = 0.03;   // 3% TP (1:2 risk/reward)
        } else {
            slPercent = 0.025;  // 2.5% SL
            tpPercent = 0.05;   // 5% TP (1:2 risk/reward)
        }

        // Adjust for confidence - higher confidence = tighter SL, wider TP
        const conf = prediction.confidence / 100;
        if (conf > 0.75) {
            tpPercent *= 1.2;   // More confident = bigger target
        } else if (conf < 0.55) {
            slPercent *= 0.8;   // Less confident = tighter stop
            tpPercent *= 0.8;
        }

        if (prediction.signal === 'LONG') {
            stopLoss = entryPrice * (1 - slPercent);
            takeProfit = entryPrice * (1 + tpPercent);
        } else if (prediction.signal === 'SHORT') {
            stopLoss = entryPrice * (1 + slPercent);
            takeProfit = entryPrice * (1 - tpPercent);
        }

        // confidence from API is already a percentage (e.g. 39.06 = 39.06%)
        const signalData = {
            signal: prediction.signal,
            confidence: prediction.confidence / 100,  // Store as decimal for display
            entryPrice,
            stopLoss,
            takeProfit,
            coin: selectedCoin,
            timeframe: displayTimeframe,
            timestamp: new Date().toISOString(),
            timestampFormatted: new Date().toLocaleString(),
            reason: prediction.reason || ''
        };

        // Save to history
        if (prediction.signal !== 'NO TRADE') {
            signalHistory.unshift(signalData);
            localStorage.setItem('alpina_signal_history', JSON.stringify(signalHistory));
        }

        // Increment attempts (only for non-admins, and only for real signals - NOT for NO TRADE)
        if (!isAdmin() && prediction.signal !== 'NO TRADE') {
            attemptsUsed++;
            saveAttempts();
            updateAttemptsDisplay();

            if (attemptsUsed >= FREE_ATTEMPTS_LIMIT) {
                setTimeout(() => {
                    showSubscriptionModal();
                }, 3000);
            }
        }

        // Show result
        showSignalResult(signalData);

        // Haptic feedback
        if (tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('success');
        }

    } catch (error) {
        console.error('Error getting signal:', error);
        hideLoading();

        // Show actual server error for debugging
        let errorMsg = error.message || 'Failed to get signal. Please try again.';
        if (errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError')) {
            errorMsg = 'Server is restarting. Please wait 1-2 minutes.';
        } else if (errorMsg.includes('::')) {
            // Show actual server error detail
            errorMsg = errorMsg.split('::').slice(1).join('::');
        }
        showErrorResult(errorMsg);

        if (tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('error');
        }
    }
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
function showErrorResult(message = 'Connection error. Please check your internet and try again.') {
    const resultBox = document.getElementById('resultBox');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const resultSubtext = document.getElementById('resultSubtext');
    const slTpBox = document.getElementById('slTpBox');

    resultBox.classList.remove('success');
    resultBox.classList.add('error');
    resultIcon.textContent = '⚠️';
    resultText.textContent = 'Error';
    resultSubtext.textContent = message;
    slTpBox.style.display = 'none';
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
function hideAllContent() {
    document.getElementById('signalsContent').style.display = 'none';
    document.getElementById('historyContent').style.display = 'none';
    document.getElementById('aboutContent').style.display = 'none';
    document.getElementById('adminContent').style.display = 'none';
}

function setActiveNavBtn(clickedBtn) {
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    if (clickedBtn) clickedBtn.classList.add('active');
}

function showSignalsTab() {
    setActiveNavBtn(event?.target);
    hideAllContent();
    document.getElementById('signalsContent').style.display = 'block';
}

function showHistoryTab() {
    setActiveNavBtn(event?.target);
    hideAllContent();
    document.getElementById('historyContent').style.display = 'block';
    updateHistoryDisplay();
}

function showAboutTab() {
    setActiveNavBtn(event?.target);
    hideAllContent();
    document.getElementById('aboutContent').style.display = 'block';
}

// Admin Panel Functions
function showAdminTab() {
    if (!isAdmin()) {
        alert('Access denied. Admin only.');
        return;
    }

    setActiveNavBtn(event?.target);
    hideAllContent();
    document.getElementById('adminContent').style.display = 'block';

    // Load admin stats
    loadAdminStats();
}

async function loadAdminStats() {
    if (!isAdmin()) return;

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/admin/stats`, {
            method: 'GET',
            headers: {
                'X-Telegram-User-ID': userId.toString()
            }
        });

        const data = await response.json();
        hideLoading();

        if (data.error) {
            alert('Error loading stats: ' + data.error);
            return;
        }

        // Update overview stats
        document.getElementById('adminTotalUsers').textContent = data.total_users || 0;
        document.getElementById('adminActiveUsers').textContent = data.active_users_7d || 0;
        document.getElementById('adminFree').textContent = data.tier_counts?.free || 0;
        document.getElementById('adminBasic').textContent = data.tier_counts?.basic || 0;
        document.getElementById('adminPro').textContent = data.tier_counts?.pro || 0;
        document.getElementById('adminPremium').textContent = data.tier_counts?.premium || 0;

        // Update recent users list
        const recentList = document.getElementById('adminRecentList');
        const emptyState = document.getElementById('adminEmpty');
        const recentCount = document.getElementById('adminRecentCount');

        if (data.recent_users && data.recent_users.length > 0) {
            recentList.style.display = 'flex';
            emptyState.style.display = 'none';
            recentCount.textContent = `${data.recent_users.length} users`;

            recentList.innerHTML = '';

            data.recent_users.forEach(user => {
                const tierColor = {
                    'free': 'var(--text-muted)',
                    'basic': 'var(--yellow)',
                    'pro': 'var(--success)',
                    'premium': 'var(--error)'
                };

                const item = document.createElement('div');
                item.style.cssText = 'padding: 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px;';
                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <div>
                            <div style="font-size: 14px; font-weight: 600;">${user.first_name}</div>
                            <div style="font-size: 12px; color: var(--text-muted);">@${user.username}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; color: ${tierColor[user.tier]}; font-weight: 700; text-transform: uppercase;">${user.tier}</div>
                            <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">${user.created_at}</div>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted);">
                        Telegram ID: ${user.telegram_id}
                    </div>
                `;
                recentList.appendChild(item);
            });
        } else {
            recentList.style.display = 'none';
            emptyState.style.display = 'block';
            recentCount.textContent = '0 users';
        }

    } catch (error) {
        hideLoading();
        console.error('Error loading admin stats:', error);
        alert('Failed to load admin statistics');
    }
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
window.showAdminTab = showAdminTab;
window.loadAdminStats = loadAdminStats;
