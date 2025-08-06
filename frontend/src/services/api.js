// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000/api'

// API Service for chat functionality
class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL
    this.conversationId = null
  }

  // Set conversation ID for current session
  setConversationId(id) {
    this.conversationId = id
  }

  // Get conversation ID
  getConversationId() {
    return this.conversationId
  }

  // Create new conversation
  async createConversation() {
    try {
      const response = await fetch(`${this.baseURL}/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      this.conversationId = data.conversationId
      return data
    } catch (error) {
      console.error('Error creating conversation:', error)
      throw error
    }
  }

  // Load conversation history
  async loadConversationHistory(conversationId = null) {
    try {
      const id = conversationId || this.conversationId
      if (!id) {
        throw new Error('No conversation ID provided')
      }

      const response = await fetch(`${this.baseURL}/conversations/${id}/messages`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data.messages || []
    } catch (error) {
      console.error('Error loading conversation history:', error)
      throw error
    }
  }

  // Send message with streaming response
  async sendMessage(message, files = [], onChunk = null, onComplete = null) {
    try {
      const formData = new FormData()
      formData.append('message', message)
      
      // Add files if any
      if (files && files.length > 0) {
        files.forEach((file, index) => {
          formData.append(`files`, file)
        })
      }

      // Add conversation ID if exists
      if (this.conversationId) {
        formData.append('conversationId', this.conversationId)
      }

      const response = await fetch(`${this.baseURL}/chat`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Handle streaming response
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullResponse = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            
            if (data === '[DONE]') {
              if (onComplete) onComplete(fullResponse)
              return fullResponse
            }

            try {
              const parsed = JSON.parse(data)
              if (parsed.content) {
                fullResponse += parsed.content
                if (onChunk) onChunk(parsed.content, fullResponse)
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }

      if (onComplete) onComplete(fullResponse)
      return fullResponse
    } catch (error) {
      console.error('Error sending message:', error)
      throw error
    }
  }

  // Upload file
  async uploadFile(file) {
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${this.baseURL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error uploading file:', error)
      throw error
    }
  }

  // Get conversations list
  async getConversations() {
    try {
      const response = await fetch(`${this.baseURL}/conversations`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data.conversations || []
    } catch (error) {
      console.error('Error getting conversations:', error)
      throw error
    }
  }

  // Delete conversation
  async deleteConversation(conversationId) {
    try {
      const response = await fetch(`${this.baseURL}/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return true
    } catch (error) {
      console.error('Error deleting conversation:', error)
      throw error
    }
  }

  // Edit message
  async editMessage(messageId, newContent) {
    try {
      const response = await fetch(`${this.baseURL}/messages/${messageId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: newContent,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error editing message:', error)
      throw error
    }
  }
}

// Create singleton instance
const apiService = new ApiService()

export default apiService 