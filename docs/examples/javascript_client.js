/**
 * CyberPuppy API JavaScript Client SDK
 * 中文網路霸凌防治 API JavaScript 客戶端
 *
 * 功能特色:
 * - 支援 Node.js 和瀏覽器環境
 * - 自動重試與錯誤處理
 * - 限流管理
 * - Promise 和 async/await 支援
 * - TypeScript 類型定義
 * - 詳細的錯誤分類
 */

// 環境檢測
const isNode = typeof window === 'undefined';
const fetch = isNode ? require('node-fetch') : window.fetch;

/**
 * 客戶端配置
 */
class ClientConfig {
  constructor({
    apiKey,
    baseUrl = 'https://api.cyberpuppy.ai',
    timeout = 30000,
    maxRetries = 3,
    retryDelay = 1000,
    rateLimit = 30
  }) {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.timeout = timeout;
    this.maxRetries = maxRetries;
    this.retryDelay = retryDelay;
    this.rateLimit = rateLimit;
  }
}

/**
 * 分析結果類別
 */
class AnalysisResult {
  constructor(data) {
    this.toxicity = data.toxicity;
    this.bullying = data.bullying;
    this.role = data.role;
    this.emotion = data.emotion;
    this.emotionStrength = data.emotion_strength;
    this.scores = data.scores;
    this.explanations = data.explanations;
    this.textHash = data.text_hash;
    this.timestamp = data.timestamp;
    this.processingTimeMs = data.processing_time_ms;
    this.rawResponse = data;
  }
}

/**
 * 錯誤類別
 */
class CyberPuppyError extends Error {
  constructor(message, statusCode = null) {
    super(message);
    this.name = 'CyberPuppyError';
    this.statusCode = statusCode;
  }
}

class AuthenticationError extends CyberPuppyError {
  constructor(message = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends CyberPuppyError {
  constructor(message = 'Rate limit exceeded') {
    super(message, 429);
    this.name = 'RateLimitError';
  }
}

class ValidationError extends CyberPuppyError {
  constructor(message = 'Validation failed') {
    super(message, 400);
    this.name = 'ValidationError';
  }
}

class ServiceError extends CyberPuppyError {
  constructor(message = 'Service error', statusCode = 500) {
    super(message, statusCode);
    this.name = 'ServiceError';
  }
}

/**
 * CyberPuppy API 客戶端
 */
class CyberPuppyClient {
  constructor(config) {
    if (!config.apiKey) {
      throw new ValidationError('API key is required');
    }

    this.config = config;
    this.requestTimes = [];
  }

  /**
   * 檢查限流狀態
   */
  _checkRateLimit() {
    const now = Date.now();
    // 移除一分鐘前的請求記錄
    this.requestTimes = this.requestTimes.filter(time => now - time < 60000);

    if (this.requestTimes.length >= this.config.rateLimit) {
      throw new RateLimitError(
        `Rate limit exceeded: ${this.config.rateLimit} requests per minute`
      );
    }

    this.requestTimes.push(now);
  }

  /**
   * 處理 API 錯誤回應
   */
  _handleError(response, responseData) {
    if (response.status === 401) {
      throw new AuthenticationError('Invalid API key');
    } else if (response.status === 429) {
      throw new RateLimitError('Rate limit exceeded');
    } else if (response.status === 400) {
      const message = responseData?.message || 'Validation error';
      throw new ValidationError(message);
    } else if (response.status >= 500) {
      throw new ServiceError(`Service error: ${response.status}`, response.status);
    } else {
      throw new CyberPuppyError(
        `Unexpected error: ${response.status}`,
        response.status
      );
    }
  }

  /**
   * 發送 API 請求（含重試機制）
   */
  async _makeRequest(method, endpoint, data = null) {
    this._checkRateLimit();

    const url = `${this.config.baseUrl}${endpoint}`;
    const options = {
      method,
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'CyberPuppy-JavaScript-SDK/1.0.0'
      }
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    // 設定超時
    if (this.config.timeout) {
      const controller = new AbortController();
      options.signal = controller.signal;
      setTimeout(() => controller.abort(), this.config.timeout);
    }

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        console.log(`Making request to ${url} (attempt ${attempt + 1})`);
        const response = await fetch(url, options);

        let responseData;
        try {
          responseData = await response.json();
        } catch (e) {
          responseData = null;
        }

        if (response.ok) {
          return responseData;
        }

        // 如果是客戶端錯誤，不重試
        if (response.status >= 400 && response.status < 500) {
          this._handleError(response, responseData);
        }

        // 服務器錯誤，進行重試
        if (attempt === this.config.maxRetries - 1) {
          this._handleError(response, responseData);
        }

      } catch (error) {
        // 如果是網路錯誤且不是最後一次嘗試，則重試
        if (attempt === this.config.maxRetries - 1) {
          if (error.name === 'AbortError') {
            throw new ServiceError('Request timeout');
          } else if (error instanceof CyberPuppyError) {
            throw error;
          } else {
            throw new ServiceError(`Request failed: ${error.message}`);
          }
        }

        console.warn(`Request failed (attempt ${attempt + 1}):`, error.message);
      }

      // 指數退避
      if (attempt < this.config.maxRetries - 1) {
        const delay = this.config.retryDelay * Math.pow(2, attempt);
        console.log(`Retrying in ${delay}ms...`);
        await this._sleep(delay);
      }
    }

    throw new ServiceError('Max retries exceeded');
  }

  /**
   * 等待指定毫秒
   */
  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 輸入驗證
   */
  _validateInput(text, context = null, threadId = null) {
    if (!text || !text.trim()) {
      throw new ValidationError('Text cannot be empty');
    }
    if (text.length > 1000) {
      throw new ValidationError('Text exceeds maximum length of 1000 characters');
    }
    if (context && context.length > 2000) {
      throw new ValidationError('Context exceeds maximum length of 2000 characters');
    }
    if (threadId && threadId.length > 50) {
      throw new ValidationError('Thread ID exceeds maximum length of 50 characters');
    }
  }

  /**
   * 生成文本雜湊（用於日誌）
   */
  _generateTextHash(text) {
    // 簡單的雜湊函數（僅用於示例，實際應用建議使用加密庫）
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // 轉為 32-bit 整數
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  /**
   * 分析單一文本
   *
   * @param {string} text - 待分析文本 (1-1000 字符)
   * @param {string|null} context - 對話上下文 (可選，最多 2000 字符)
   * @param {string|null} threadId - 對話串 ID (可選，最多 50 字符)
   * @returns {Promise<AnalysisResult>} 分析結果
   */
  async analyzeText(text, context = null, threadId = null) {
    // 輸入驗證
    this._validateInput(text, context, threadId);

    const payload = { text: text.trim() };
    if (context) {
      payload.context = context.trim();
    }
    if (threadId) {
      payload.thread_id = threadId.trim();
    }

    const textHash = this._generateTextHash(text);
    console.log(`Analyzing text (length: ${text.length}, hash: ${textHash})`);

    const response = await this._makeRequest('POST', '/analyze', payload);
    return new AnalysisResult(response);
  }

  /**
   * 批次分析多個文本
   *
   * @param {string[]} texts - 文本列表
   * @param {string[]|null} contexts - 對應的上下文列表 (可選)
   * @param {string[]|null} threadIds - 對應的對話串 ID 列表 (可選)
   * @param {number} maxConcurrent - 最大並行請求數
   * @returns {Promise<AnalysisResult[]>} 分析結果列表
   */
  async analyzeBatch(texts, contexts = null, threadIds = null, maxConcurrent = 5) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new ValidationError('Text list cannot be empty');
    }

    console.log(`Starting batch analysis of ${texts.length} texts`);
    const startTime = Date.now();

    // 建立並行控制器
    const semaphore = new Semaphore(maxConcurrent);

    // 建立任務列表
    const tasks = texts.map(async (text, index) => {
      return semaphore.acquire(async () => {
        const context = contexts && index < contexts.length ? contexts[index] : null;
        const threadId = threadIds && index < threadIds.length ? threadIds[index] : null;

        try {
          return await this.analyzeText(text, context, threadId);
        } catch (error) {
          return { error: error.message, index };
        }
      });
    });

    // 執行所有任務
    const results = await Promise.all(tasks);

    const elapsedTime = Date.now() - startTime;
    console.log(`Batch analysis completed in ${elapsedTime}ms`);

    // 分離成功和失敗的結果
    const successfulResults = results.filter(result => !(result.error));
    const failedResults = results.filter(result => result.error);

    if (failedResults.length > 0) {
      console.warn(`Failed to analyze ${failedResults.length} texts:`, failedResults);
    }

    return successfulResults;
  }

  /**
   * 檢查 API 服務狀態
   *
   * @returns {Promise<Object>} 健康檢查結果
   */
  async healthCheck() {
    return await this._makeRequest('GET', '/healthz');
  }

  /**
   * 取得限流狀態
   *
   * @returns {Object} 限流狀態資訊
   */
  getRateLimitStatus() {
    const now = Date.now();
    const recentRequests = this.requestTimes.filter(time => now - time < 60000);

    return {
      requestsInLastMinute: recentRequests.length,
      rateLimit: this.config.rateLimit,
      remaining: Math.max(0, this.config.rateLimit - recentRequests.length),
      resetTime: recentRequests.length > 0 ? Math.max(...recentRequests) + 60000 : now
    };
  }
}

/**
 * 簡單的信號量實現（用於並行控制）
 */
class Semaphore {
  constructor(max) {
    this.max = max;
    this.count = 0;
    this.queue = [];
  }

  async acquire(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this._tryNext();
    });
  }

  _tryNext() {
    if (this.count < this.max && this.queue.length > 0) {
      this.count++;
      const { task, resolve, reject } = this.queue.shift();

      task()
        .then(resolve)
        .catch(reject)
        .finally(() => {
          this.count--;
          this._tryNext();
        });
    }
  }
}

// 使用範例
async function exampleUsage() {
  console.log('=== CyberPuppy JavaScript SDK 範例 ===');

  // 配置客戶端
  const config = new ClientConfig({
    apiKey: 'cp_your_api_key_here',
    baseUrl: 'https://api.cyberpuppy.ai',
    timeout: 30000,
    maxRetries: 3,
    rateLimit: 30
  });

  const client = new CyberPuppyClient(config);

  try {
    // 健康檢查
    const health = await client.healthCheck();
    console.log(`API 狀態: ${health.status}`);

    // 單一文本分析
    const result = await client.analyzeText('你好，今天天氣真好！');
    console.log(`毒性等級: ${result.toxicity}`);
    console.log(`情緒: ${result.emotion} (強度: ${result.emotionStrength})`);
    console.log(`重要詞彙:`, result.explanations.important_words);

    // 帶上下文的分析
    const contextResult = await client.analyzeText(
      '我不同意你的看法',
      '剛才討論的是關於教育政策的議題',
      'edu_discussion_001'
    );
    console.log(`上下文分析結果: ${contextResult.toxicity}`);

    // 批次分析
    const texts = [
      '你好嗎？',
      '今天天氣很好',
      '你這個笨蛋！',
      '謝謝你的幫助'
    ];

    const batchResults = await client.analyzeBatch(texts, null, null, 2);
    console.log(`批次分析完成，處理了 ${batchResults.length} 個文本`);

    batchResults.forEach((result, index) => {
      console.log(`文本 ${index}: ${result.toxicity}`);
    });

    // 限流狀態
    const rateStatus = client.getRateLimitStatus();
    console.log(`剩餘請求次數: ${rateStatus.remaining}`);

  } catch (error) {
    if (error instanceof ValidationError) {
      console.error(`輸入驗證錯誤: ${error.message}`);
    } else if (error instanceof AuthenticationError) {
      console.error(`認證錯誤: ${error.message}`);
    } else if (error instanceof RateLimitError) {
      console.error(`限流錯誤: ${error.message}`);
    } else if (error instanceof ServiceError) {
      console.error(`服務錯誤: ${error.message}`);
    } else {
      console.error(`未知錯誤: ${error.message}`);
    }
  }
}

// Node.js 環境導出
if (isNode) {
  module.exports = {
    CyberPuppyClient,
    ClientConfig,
    AnalysisResult,
    CyberPuppyError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServiceError
  };
}

// 瀏覽器環境全域變數
if (typeof window !== 'undefined') {
  window.CyberPuppy = {
    Client: CyberPuppyClient,
    ClientConfig,
    AnalysisResult,
    CyberPuppyError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServiceError
  };
}

// 範例執行
if (isNode && require.main === module) {
  exampleUsage().catch(console.error);
}

/**
 * TypeScript 類型定義（參考用）
 *
 * interface ClientConfigOptions {
 *   apiKey: string;
 *   baseUrl?: string;
 *   timeout?: number;
 *   maxRetries?: number;
 *   retryDelay?: number;
 *   rateLimit?: number;
 * }
 *
 * interface AnalysisResult {
 *   toxicity: 'none' | 'toxic' | 'severe';
 *   bullying: 'none' | 'harassment' | 'threat';
 *   role: 'none' | 'perpetrator' | 'victim' | 'bystander';
 *   emotion: 'pos' | 'neu' | 'neg';
 *   emotionStrength: number;
 *   scores: {
 *     toxicity: Record<string, number>;
 *     bullying: Record<string, number>;
 *     role: Record<string, number>;
 *     emotion: Record<string, number>;
 *   };
 *   explanations: {
 *     important_words: Array<{word: string, importance: number}>;
 *     method: string;
 *     confidence: number;
 *   };
 *   textHash: string;
 *   timestamp: string;
 *   processingTimeMs: number;
 * }
 *
 * interface HealthCheckResult {
 *   status: string;
 *   timestamp: string;
 *   version: string;
 *   uptime_seconds: number;
 * }
 */