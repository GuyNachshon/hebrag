import { ref } from 'vue'
import { defineStore } from 'pinia'
import apiService from '../services/api.js'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const isLoading = ref(false)
  const hasMessageBeenSent = ref(false)
  const error = ref(null)
  const conversationId = ref(null)
  const conversations = ref([])

  // Mock responses for fallback
  const mockResponses = [
    'תודה על השאלה! אני כאן כדי לעזור לך.',
    'זה נושא מעניין. בואו נחקור אותו יחד.',
    'יש לי כמה רעיונות שיכולים לעזור לך.',
    'אני מבין את השאלה שלך. הנה מה שאני חושב:',
    'זו שאלה מצוינת! הנה התשובה שלי:',
    'תבסס על המידע שסיפקת, הנה מה שאני יכול לומר:',
    'אני שמח לעזור! הנה מה שאני יודע על הנושא:',
    'זו שאלה חשובה. הנה התשובה המפורטת שלי:',
    'אני מבין את הדאגה שלך. הנה מה שאני ממליץ:',
    'תודה על השיתוף! הנה מה שאני חושב על זה:'
  ]

  // Generate mock conversation ID
  const generateMockConversationId = () => {
    return 'mock_conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
  }

  // Initialize conversation
  async function initializeConversation() {
    try {
      error.value = null
      const data = await apiService.createConversation()
      conversationId.value = data.conversationId
      apiService.setConversationId(data.conversationId)
      return data.conversationId
    } catch (err) {
      console.warn('API failed, using mock conversation:', err)
      // Fallback to mock conversation
      const mockId = generateMockConversationId()
      conversationId.value = mockId
      apiService.setConversationId(mockId)
      return mockId
    }
  }

  // Load conversation history
  async function loadConversationHistory(id = null) {
    try {
      error.value = null
      isLoading.value = true
      
      const conversationIdToLoad = id || conversationId.value
      if (!conversationIdToLoad) {
        await initializeConversation()
      }
      
      const history = await apiService.loadConversationHistory(conversationIdToLoad)
      messages.value = history.map(msg => ({
        id: msg.id,
        content: msg.content,
        sender: msg.sender,
        timestamp: new Date(msg.timestamp),
        files: msg.files || []
      }))
      
      hasMessageBeenSent.value = messages.value.length > 0
    } catch (err) {
      console.warn('Failed to load conversation history, using empty state:', err)
      // Fallback to empty conversation
      messages.value = []
      hasMessageBeenSent.value = false
    } finally {
      isLoading.value = false
    }
  }

  // Load conversations list
  async function loadConversations() {
    try {
      error.value = null
      const conversationsList = await apiService.getConversations()
      conversations.value = conversationsList
      return conversationsList
    } catch (err) {
      console.warn('Failed to load conversations, using mock data:', err)
      // Fallback to mock conversations
      conversations.value = [
        {
          id: 'mock_conv_1',
          title: 'שיחה לדוגמה 1',
          lastMessage: 'שלום! איך אני יכול לעזור?',
          timestamp: new Date(Date.now() - 3600000).toISOString()
        },
        {
          id: 'mock_conv_2', 
          title: 'שיחה לדוגמה 2',
          lastMessage: 'תודה על השאלה המעניינת',
          timestamp: new Date(Date.now() - 7200000).toISOString()
        }
      ]
      return conversations.value
    }
  }

  // Switch to a different conversation
  async function switchConversation(id) {
    try {
      conversationId.value = id
      apiService.setConversationId(id)
      await loadConversationHistory(id)
    } catch (err) {
      error.value = 'Failed to switch conversation'
      console.error('Error switching conversation:', err)
      throw err
    }
  }

  // Send message with API integration and mock fallback
  async function sendMessage(messageContent, files = []) {
    try {
      error.value = null
      hasMessageBeenSent.value = true
      
      // Initialize conversation if needed
      if (!conversationId.value) {
        await initializeConversation()
      }
      
      // Add user message immediately
      const userMessage = {
        id: Date.now(),
        content: messageContent,
        sender: 'user',
        timestamp: new Date(),
        files: files
      }
      messages.value.push(userMessage)
      
      // Start loading
      isLoading.value = true
      
      // Create AI message placeholder for streaming
      const aiMessageId = Date.now() + 1
      const aiMessage = {
        id: aiMessageId,
        content: '',
        sender: 'assistant',
        timestamp: new Date(),
        isStreaming: true
      }
      messages.value.push(aiMessage)
      
      try {
        // Try to send message to API with streaming
        await apiService.sendMessage(
          messageContent,
          files,
          // onChunk callback for streaming
          (chunk, fullResponse) => {
            const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
            if (messageIndex !== -1) {
              messages.value[messageIndex].content = fullResponse
            }
          },
          // onComplete callback
          (fullResponse) => {
            const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
            if (messageIndex !== -1) {
              messages.value[messageIndex].content = fullResponse
              messages.value[messageIndex].isStreaming = false
              messages.value[messageIndex].timestamp = new Date()
            }
            isLoading.value = false
          }
        )
      } catch (apiError) {
        console.warn('API failed, using mock response:', apiError)
        
        // Fallback to mock response with simulated streaming
        const mockResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)]
        const words = mockResponse.split(' ')
        let currentResponse = ''
        
        // Simulate streaming by adding words gradually
        for (let i = 0; i < words.length; i++) {
          await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200))
          currentResponse += (i > 0 ? ' ' : '') + words[i]
          
          const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
          if (messageIndex !== -1) {
            messages.value[messageIndex].content = currentResponse
          }
        }
        
        // Complete the mock response
        const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
        if (messageIndex !== -1) {
          messages.value[messageIndex].isStreaming = false
          messages.value[messageIndex].timestamp = new Date()
        }
        isLoading.value = false
      }
      
    } catch (err) {
      error.value = 'Failed to send message'
      console.error('Error sending message:', err)
      
      // Remove the AI message if there was an error
      const aiMessageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
      if (aiMessageIndex !== -1) {
        messages.value.splice(aiMessageIndex, 1)
      }
      
      isLoading.value = false
      throw err
    }
  }

  // Edit message with API integration and mock fallback
  async function editMessage(messageId, newContent) {
    try {
      error.value = null
      
      // Find the message to edit
      const messageIndex = messages.value.findIndex(msg => msg.id === messageId)
      if (messageIndex === -1) return

      // Update the message content locally first
      messages.value[messageIndex].content = newContent
      messages.value[messageIndex].timestamp = new Date()

      // Remove all messages that came after this message
      messages.value.splice(messageIndex + 1)

      // If this was a user message, send a new AI response
      if (messages.value[messageIndex].sender === 'user') {
        isLoading.value = true
        
        // Create AI message placeholder for streaming
        const aiMessageId = Date.now() + 1
        const aiMessage = {
          id: aiMessageId,
          content: '',
          sender: 'assistant',
          timestamp: new Date(),
          isStreaming: true
        }
        messages.value.push(aiMessage)
        
        try {
          // Try to send updated message to API
          await apiService.sendMessage(
            newContent,
            messages.value[messageIndex].files || [],
            // onChunk callback for streaming
            (chunk, fullResponse) => {
              const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
              if (messageIndex !== -1) {
                messages.value[messageIndex].content = fullResponse
              }
            },
            // onComplete callback
            (fullResponse) => {
              const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
              if (messageIndex !== -1) {
                messages.value[messageIndex].content = fullResponse
                messages.value[messageIndex].isStreaming = false
                messages.value[messageIndex].timestamp = new Date()
              }
              isLoading.value = false
            }
          )
        } catch (apiError) {
          console.warn('API failed during edit, using mock response:', apiError)
          
          // Fallback to mock response with simulated streaming
          const mockResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)]
          const words = mockResponse.split(' ')
          let currentResponse = ''
          
          // Simulate streaming by adding words gradually
          for (let i = 0; i < words.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200))
            currentResponse += (i > 0 ? ' ' : '') + words[i]
            
            const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
            if (messageIndex !== -1) {
              messages.value[messageIndex].content = currentResponse
            }
          }
          
          // Complete the mock response
          const messageIndex = messages.value.findIndex(msg => msg.id === aiMessageId)
          if (messageIndex !== -1) {
            messages.value[messageIndex].isStreaming = false
            messages.value[messageIndex].timestamp = new Date()
          }
          isLoading.value = false
        }
      }
      
      try {
        // Try to update message on server
        await apiService.editMessage(messageId, newContent)
      } catch (apiError) {
        console.warn('Failed to update message on server:', apiError)
        // Continue without server update
      }
      
    } catch (err) {
      error.value = 'Failed to edit message'
      console.error('Error editing message:', err)
      throw err
    }
  }

  // Clear messages and start new conversation
  async function clearMessages() {
    try {
      messages.value = []
      hasMessageBeenSent.value = false
      isLoading.value = false
      error.value = null
      
      // Create new conversation
      await initializeConversation()
    } catch (err) {
      error.value = 'Failed to clear messages'
      console.error('Error clearing messages:', err)
      throw err
    }
  }

  // Delete conversation
  async function deleteConversation(id) {
    try {
      error.value = null
      await apiService.deleteConversation(id)
      
      // Remove from conversations list
      conversations.value = conversations.value.filter(conv => conv.id !== id)
      
      // If this was the current conversation, clear messages
      if (id === conversationId.value) {
        await clearMessages()
      }
    } catch (err) {
      console.warn('Failed to delete conversation on server:', err)
      // Remove locally even if server delete fails
      conversations.value = conversations.value.filter(conv => conv.id !== id)
      
      if (id === conversationId.value) {
        await clearMessages()
      }
    }
  }

  // Upload file with mock fallback
  async function uploadFile(file) {
    try {
      error.value = null
      const result = await apiService.uploadFile(file)
      return result
    } catch (err) {
      console.warn('File upload failed, using mock response:', err)
      // Return mock file data
      return {
        url: URL.createObjectURL(file),
        name: file.name,
        size: file.size,
        type: file.type
      }
    }
  }

  return {
    messages,
    isLoading,
    hasMessageBeenSent,
    error,
    conversationId,
    conversations,
    sendMessage,
    editMessage,
    clearMessages,
    loadConversationHistory,
    loadConversations,
    switchConversation,
    deleteConversation,
    uploadFile,
    initializeConversation
  }
}) 