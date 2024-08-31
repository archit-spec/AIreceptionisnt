const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

const socket = new WebSocket('ws://localhost:8000/ws');

socket.onmessage = function(event) {
    addMessage(event.data, 'ai');
};

chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const message = userInput.value;
    addMessage(message, 'user');
    socket.send(message);
    userInput.value = '';
});

function addMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Initial greeting
addMessage("Welcome to Dr. Adrin's office. Are you having an emergency or would you like to leave a message?", 'ai');