import asyncio
import json
import logging
import os
import random
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
import aiosqlite
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import NeuralNet
from nlp import tokenize, lemmatize, normalize_text, bag_of_words
from qa import questions_seedballs, microcourse_questions_seedballs

# Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "sessions.db")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
db_connection = None

# FastAPI app


# CORS middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connection
    
    # Startup
    db_connection = await aiosqlite.connect(DATABASE_PATH)
    await db_connection.execute('PRAGMA journal_mode=WAL')
    await db_connection.execute('PRAGMA synchronous=NORMAL')
    await db_connection.execute('PRAGMA cache_size=10000')
    await db_connection.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_name TEXT,
            quiz_state TEXT,
            current_quiz TEXT,
            current_question INTEGER,
            main_quiz_correct INTEGER,
            microcourse_correct INTEGER
        )
    ''')
    await db_connection.commit()
    
    yield
    
    # Shutdown
    if db_connection:
        await db_connection.close()
app = FastAPI(lifespan=lifespan)
app.add_event_handler("startup", lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

async def load_intents():
    async with aiofiles.open('intents.json', mode='r') as f:
        return json.loads(await f.read())

intents = asyncio.run(load_intents())

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Greek"

# Helper functions
async def get_db_connection():
    return db_connection

async def get_user_state(session_id: str, db):
    async with db.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,)) as cursor:
        row = await cursor.fetchone()
        if row:
            return dict(zip([column[0] for column in cursor.description], row))
    return None

async def update_user_state(session_id: str, db, **kwargs):
    query = 'UPDATE sessions SET '
    values = []
    for key, value in kwargs.items():
        query += f'{key} = ?, '
        values.append(value)
    query = query.rstrip(', ') + ' WHERE session_id = ?'
    values.append(session_id)
    await db.execute(query, values)
    await db.commit()

async def create_user_state(session_id: str, db):
    await db.execute('''
        INSERT INTO sessions (session_id, quiz_state, current_quiz, current_question, main_quiz_correct, microcourse_correct)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, 'inactive', '', 0, 0, 0))
    await db.commit()

async def start_quiz(session_id: str, quiz_type: str, db):
    user_state = await get_user_state(session_id, db)
    
    if not user_state:
        await create_user_state(session_id, db)
        user_state = await get_user_state(session_id, db)

    if user_state['quiz_state'] == 'inactive' or user_state['current_quiz'] != quiz_type:
        await update_user_state(session_id, db, main_quiz_correct=0, microcourse_correct=0)
    
    await update_user_state(session_id, db, quiz_state='active', current_quiz=quiz_type, current_question=0)
    
    if not user_state.get('user_name'):
        return "Before we start, I'd love to know your name. What should I call you?"
    
    quiz_intro = random.choice([
        f"Alright {user_state['user_name']}, let's dive into the world of seed balls! Ready for a fun challenge?",
        f"Hey {user_state['user_name']}, excited to test your seed ball knowledge? Let's get started!",
        f"OK {user_state['user_name']}, time to put on your thinking cap. This quiz is all about seed balls!"
    ])
    
    return quiz_intro + "\n\n" + await get_next_question(session_id, db)

async def get_next_question(session_id: str, db):
    user_state = await get_user_state(session_id, db)
    current_question_index = user_state['current_question']
    
    current_quiz = questions_seedballs if user_state['current_quiz'] == "main" else microcourse_questions_seedballs
    
    if current_question_index < len(current_quiz):
        q = current_quiz[current_question_index]
        question_intro = random.choice([
            f"Question {current_question_index + 1} coming up!",
            f"Here's number {current_question_index + 1} for you.",
            f"Let's see how you handle this one:",
            f"Question {current_question_index + 1} - give it your best shot!",
        ])
        question_text = f"{question_intro}\n\n{q['question']}"
        
        return question_text
    else:
        return await end_quiz(session_id, db)

async def check_quiz_answer(session_id: str, user_answer: str, db):
    user_state = await get_user_state(session_id, db)
    current_quiz = questions_seedballs if user_state['current_quiz'] == "main" else microcourse_questions_seedballs
    current_question_index = user_state['current_question']
    q = current_quiz[current_question_index]
    
    if user_answer.lower() == q["answer"].lower():
        correct_response = random.choice([
            f"That's right!, {user_state['user_name']}! vola!\n\n",
            f"you are right {user_state['user_name']}! You nailed it. vola!\n\n"
        ])
        result = correct_response + "\n"
        if user_state['current_quiz'] == "main":
            await update_user_state(session_id, db, main_quiz_correct=user_state['main_quiz_correct'] + 1)
        else:
            await update_user_state(session_id, db, microcourse_correct=user_state['microcourse_correct'] + 1)
    else:
        incorrect_response = random.choice([
            f"thats wrong, {user_state['user_name']}. But don't worry, we're all learning here!\n\n",
            f"thats wrong, but not exactly. Keep your chin up, {user_state['user_name']}!\n\n",
            f"That's wrong, but good try, {user_state['user_name']}!\n\n",
            f"Oops, That's wrong, But I like your thinking, {user_state['user_name']}!\n\n"
        ])
        result = f"{incorrect_response} The correct answer is: {q['answer']}\n\n"
    
    result += f"Here's why: {q['explanation']}\n\n"
    
    if 'pdf_link' in q:
        result += f"For more information, check out this PDF: {q['pdf_link']}\n\n"
    
    await update_user_state(session_id, db, current_question=current_question_index + 1)
    
    if current_question_index + 1 < len(current_quiz):
        transition = random.choice([
            "Ready for the next one?",
            "Let's keep the momentum going!",
            "On to the next question!",
            "Here comes another one for you!"
        ])
        result += f"{transition}\n\n" + await get_next_question(session_id, db)
    else:
        result += await end_quiz(session_id, db)
    
    return result

def assess_understanding(correct_answers, total_questions):
    percentage = (correct_answers / total_questions) * 100
    if percentage >= 90:
        return random.choice([
            "Wow! You've got an excellent grasp of seed balls. You're practically an expert!",
            "Incredible performance! Your understanding of seed balls is top-notch.",
            "Outstanding! You've shown a deep comprehension of seed ball concepts."
        ])
    elif percentage >= 70:
        return random.choice([
            "Great job! You have a solid understanding of seed balls.",
            "Well done! You've demonstrated a good grasp of the key concepts.",
            "Nice work! Your knowledge of seed balls is quite strong."
        ])
    elif percentage >= 50:
        return random.choice([
            "Good effort! You've got a fair understanding of seed balls, but there's room for improvement.",
            "Not bad! You've grasped some key concepts, but some areas might need a bit more attention.",
            "You're on the right track! With a bit more study, you'll become a seed ball expert in no time."
        ])
    else:
        return random.choice([
            "Thanks for taking the quiz! It seems like seed balls might be a new topic for you. Let's explore it more together!",
            "Good try! Seed balls can be tricky. How about we review some of the key points?",
            "Thanks for your effort! Seed balls might be challenging, but don't worry - we can go over the basics again if you'd like."
        ])

async def end_quiz(session_id: str, db):
    user_state = await get_user_state(session_id, db)
    current_quiz = questions_seedballs if user_state['current_quiz'] == "main" else microcourse_questions_seedballs
    correct_answers = user_state['main_quiz_correct'] if user_state['current_quiz'] == "main" else user_state['microcourse_correct']
    
    result = random.choice([
        f"And that's a wrap, {user_state['user_name']}! You've completed the {user_state['current_quiz']} quiz.\n\n",
        f"Fantastic job, {user_state['user_name']}! You've made it through the {user_state['current_quiz']} quiz.\n\n",
        f"Congratulations on finishing the {user_state['current_quiz']} quiz, {user_state['user_name']}!\n\n"
    ])
    
    result += f" You got {correct_answers} out of {len(current_quiz)} questions right.\n\n"
    
    if user_state['current_quiz'] == "main":
        result += assess_understanding(user_state['main_quiz_correct'], len(questions_seedballs))
        
        await update_user_state(session_id, db, quiz_state='completed', current_question=len(current_quiz))
        
        result += "\n\n" + random.choice([
            f"So, {user_state['user_name']}, are you up for another challenge? Want to take on the microcourse quiz? (Yes/No)",
            f"Feeling brave, {user_state['user_name']}? How about tackling the microcourse quiz next? (Yes/No)",
            f"What do you say, {user_state['user_name']}? Ready to dive deeper with the microcourse quiz? (Yes/No)"
        ])
    else:
        result += assess_understanding(user_state['microcourse_correct'], len(microcourse_questions_seedballs))
        
        await update_user_state(session_id, db, quiz_state='inactive', current_quiz='', current_question=0)
        result += "\n\n" + random.choice([
            "You've learned a lot about seed balls today. Great job!",
            "I'm impressed with your seed ball knowledge. Well done!",
            "You're becoming a real seed ball expert. Keep it up!"
        ])
    
    return result

async def get_response(session_id: str, msg: str, db):
    user_state = await get_user_state(session_id, db)

    if not user_state:
        await create_user_state(session_id, db)
        user_state = await get_user_state(session_id, db)

    if not user_state.get('user_name') and user_state['quiz_state'] == 'active':
        await update_user_state(session_id, db, user_name=msg)
        return random.choice([
            f"Great to meet you, {msg}! Let's get started with the quiz.",
            f"Thanks, {msg}! I'm excited to quiz you about seed balls.",
            f"Awesome, {msg}! Ready to test your knowledge?"
        ]) + "\n\n" + await get_next_question(session_id, db)

    if user_state['quiz_state'] == 'active':
        return await check_quiz_answer(session_id, msg, db)

    if user_state['quiz_state'] == 'completed' and user_state['current_quiz'] == "main":
        if msg.lower() == "yes":
            return await start_quiz(session_id, "microcourse", db)
        elif msg.lower() == "no":
            await update_user_state(session_id, db, quiz_state='inactive', current_quiz='', current_question=0)
            return random.choice([
                f"No problem, {user_state['user_name']}! Let me know if you need anything else!",
                f"Alright, {user_state['user_name']}. You've earned a break after that quiz. What else would you like to talk about?",
                f"Sure thing, {user_state['user_name']}. Feel free to chat with me about other topics."
            ])
        else:
            return random.choice([
                f"I didn't quite catch that, {user_state['user_name']}. Just a simple 'Yes' or 'No' will do. Want to take on the microcourse quiz?",
                f"Hmm, I'm not sure what you mean, {user_state['user_name']}. Can you please answer with 'Yes' or 'No'? Are you up for the microcourse quiz?",
                f"Let's keep it simple, {user_state['user_name']}. Just say 'Yes' if you want to try the microcourse quiz, or 'No' if you're done for now."
            ])

    if msg.lower() == "seedballs microcourse":
        return await start_quiz(session_id, "microcourse", db)
    
    if msg.lower() == "seedballs":
        return await start_quiz(session_id, "main", db)

    # Handle non-quiz interactions using the neural network
    sentence = tokenize(normalize_text(msg))
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I'm sorry, I don't understand. Can you please rephrase your question?"

class ChatInput(BaseModel):
    user_input: str
    session_id: Optional[str] = None

@app.post('/chat')
async def chat(chat_input: ChatInput, db: aiosqlite.Connection = Depends(get_db_connection)):
    try:
        user_input = chat_input.user_input
        session_id = chat_input.session_id or str(uuid.uuid4())
        
        # Get the response
        response = await get_response(session_id, user_input, db)
        
        return {'response': response, 'session_id': session_id}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    #ngrok http 5000 --domain=moccasin-light-opossum.ngrok-free.app