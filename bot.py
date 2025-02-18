
import logging
import openai
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from config import OPENAI_API_KEY, YOUR_TELEGRAM_BOT_TOKEN
from faiss_db.search import search_problem
from chatgpt_handler import generate_final_response

# Установим ключ API
openai.api_key = OPENAI_API_KEY
USER_CONTEXT = {}  # Хранение истории запросов пользователей


# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Переменные состояния
WAITING_FOR_QUESTION = False
WAITING_FOR_CONFIRMATION = False

def format_for_markdown_v2(text):
    """Форматирует текст для MarkdownV2."""
    formatted_text = (
        text.replace("**", "__")  # Преобразуем ** в __ для жирного текста
            .replace("\\n", "\n")
            .replace("\n\n", "\n")
    )
    return formatted_text


def escape_markdown_v2(text):
    """Экранирует специальные символы для MarkdownV2."""
    escape_chars = r"_[]()~`>#+-=|{}.!"
    return "".join(f"\\{char}" if char in escape_chars else char for char in text)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ответ на команду /start."""
    global WAITING_FOR_QUESTION, WAITING_FOR_CONFIRMATION
    WAITING_FOR_QUESTION = False
    WAITING_FOR_CONFIRMATION = False

    keyboard = [[InlineKeyboardButton("Задать вопрос", callback_data="ask_question")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👋 Привет! Я бот-помощник.\nНажмите на кнопку ниже, чтобы задать вопрос.",
        reply_markup=reply_markup
    )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатия на кнопки."""
    global WAITING_FOR_QUESTION, WAITING_FOR_CONFIRMATION, USER_CONTEXT
    query = update.callback_query
    await query.answer()

    if query.data == "ask_question":
        WAITING_FOR_QUESTION = True
        WAITING_FOR_CONFIRMATION = False
        await query.message.reply_text("📢 Пожалуйста, введите ваш вопрос.")

    elif query.data == "answer_received":
        WAITING_FOR_QUESTION = False
        WAITING_FOR_CONFIRMATION = False
        USER_CONTEXT.pop(update.effective_user.id, None)  # Очищаем контекст после завершения
        keyboard = [[InlineKeyboardButton("Задать вопрос", callback_data="ask_question")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("👍 Спасибо за ваш вопрос! Нажмите 'Задать вопрос', чтобы задать новый.", reply_markup=reply_markup)

    elif query.data == "clarify_question":
        # Пользователь хочет уточнить вопрос, бот снова ожидает текст
        WAITING_FOR_QUESTION = True
        WAITING_FOR_CONFIRMATION = False
        await query.message.reply_text("✍ Введите ваш уточняющий вопрос:")


def split_message(text, max_length=4096):
    """Разбивает длинное сообщение на несколько частей."""
    parts = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:]
    parts.append(text)
    return parts


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения пользователей, включая уточняющие вопросы."""
    global WAITING_FOR_QUESTION, WAITING_FOR_CONFIRMATION, USER_CONTEXT

    user_query = update.message.text  # Получаем текст запроса
    user_id = update.effective_user.id  # Получаем ID пользователя
    logging.info(f"Запрос от пользователя {user_id}: {user_query}")

    if not WAITING_FOR_QUESTION and user_id not in USER_CONTEXT:
        await update.message.reply_text("❗ Для того чтобы задать вопрос, нажмите на кнопку 'Задать вопрос'.")
        return

    if WAITING_FOR_CONFIRMATION:
        await update.message.reply_text("❗ Нажмите на кнопку 'Ответ получен', чтобы завершить предыдущий вопрос.")
        return

    WAITING_FOR_QUESTION = False
    message = await update.message.reply_text("⌛ Генерирую ответ...")

    try:
        if user_id in USER_CONTEXT:
            # Получаем прошлый контекст
            previous_query = USER_CONTEXT[user_id]["original_query"]
            previous_response = USER_CONTEXT[user_id]["previous_response"]
            previous_metadata = USER_CONTEXT[user_id]["previous_metadata"]

            # Выполняем новый поиск по базе
            logging.info(f"🔎 Выполняем новый поиск по уточняющему вопросу: {user_query}") #

            new_metadata_json = search_problem(user_query)
            new_metadata_list = json.loads(new_metadata_json).get("проблемы", [])

            logging.info(f"🛠 Найдено {len(new_metadata_list)} новых записей") #


            if not new_metadata_list:
                await update.message.reply_text("⚠️ По вашему уточнению ничего не найдено. Попробуйте переформулировать запрос.")
                await message.delete()
                return

            # Объединяем старые и новые результаты
            logging.info(f"🔄 Старые данные: {len(previous_metadata)} записей")#
            logging.info(f"🔄 Новые данные: {len(new_metadata_list)} записей")#

            # Объединяем старые и новые данные
            combined_metadata = previous_metadata + new_metadata_list

            logging.info(f"🧐 Всего записей до удаления дубликатов: {len(combined_metadata)}")#


            unique_metadata = {json.dumps(record, ensure_ascii=False): record for record in combined_metadata}.values()
            combined_metadata_list = list(unique_metadata)  # Финальный список без дубликатов

            logging.info(f"✅ Количество уникальных записей после фильтрации: {len(combined_metadata_list)}") #


            # Объединяем старый и новый ответы
            combined_query = user_query

            logging.info(f"📢 Отправляем в GPT: {len(combined_metadata_list)} записей")
            logging.info(f"📜 Итоговый вопрос для GPT: {combined_query}")

            # Генерируем новый ответ на основе объединённого контекста
            final_response, token_count = generate_final_response(combined_metadata_list, combined_query)
            formatted_response = escape_markdown_v2(final_response)

            logging.info(f"📩 Финальный ответ от GPT:\n{final_response}")


            # Обновляем контекст пользователя
            USER_CONTEXT[user_id] = {
                "original_query": previous_query,
                "previous_response": final_response,  # Сохраняем новый ответ как предыдущий
                "previous_metadata": combined_metadata_list  # Обновляем объединённый JSON
            }

        else:
            # Первый запрос пользователя: выполняем поиск и сохраняем данные
            metadata_json = search_problem(user_query)
            metadata_list = json.loads(metadata_json).get("проблемы", [])

            if not metadata_list:
                await update.message.reply_text("⚠️ По вашему запросу ничего не найдено. Попробуйте уточнить запрос.")
                await message.delete()
                return

            # Генерируем ответ
            final_response, token_count = generate_final_response(metadata_list, user_query)
            formatted_response = escape_markdown_v2(final_response)

            # Сохраняем контекст первого запроса
            USER_CONTEXT[user_id] = {
                "original_query": user_query,
                "previous_response": final_response,
                "previous_metadata": metadata_list
            }

        # Добавляем кнопки для дальнейшего взаимодействия
        keyboard = [
            [InlineKeyboardButton("Ответ получен", callback_data="answer_received")],
            [InlineKeyboardButton("Уточнить вопрос", callback_data="clarify_question")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(formatted_response.replace("**", "__"), reply_markup=reply_markup, parse_mode="MarkdownV2")
        logging.info(f"Использовано токенов: {token_count}")

        WAITING_FOR_CONFIRMATION = True
        
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text("Произошла ошибка при обработке вашего запроса. Попробуйте снова.")
        await message.delete()



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ответ на команду /help."""
    await update.message.reply_text("Введите ваш вопрос, и я постараюсь вам помочь найти решение!")


def main():
    """Основная функция для запуска Telegram-бота."""
    application = ApplicationBuilder().token(YOUR_TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()