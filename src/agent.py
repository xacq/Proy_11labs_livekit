import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.voice import Agent as VoiceAgent
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.elevenlabs import TTS

#ASSEMBLYAI_API_KEY=os.environ.get("ASSEMBLYAI_API_KEY")
DEEPGRAM_API_KEY=os.environ.get("DEEPGRAM_API_KEY")

try:
    from livekit.plugins import noise_cancellation
except ImportError:  # plugin opcional
    noise_cancellation = None

logger = logging.getLogger("agent")

env_file = Path(__file__).resolve().parent.parent / ".env.local"
load_dotenv(env_file)


# --- Función auxiliar para cargar archivos de texto ---
def load_prompt(file_name: str) -> str:
    base_path = Path(__file__).resolve().parent / "prompts"
    try:
        with open(base_path / file_name, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


background_knowledge = f"""
    Usá la siguiente información solo para responder preguntas de fútbol o sobre tu carrera.
    Respondé siempre en tono entusiasta y humano.
    Nunca cites ni leas literalmente los textos, integralos en tus respuestas con naturalidad.
    {load_prompt("biografia.txt")}
    {load_prompt("datos_futbol.txt")}
    {load_prompt("phrases.txt")}
    """

# --- Clase principal del asistente ---
class Assistant(VoiceAgent):
    def __init__(self) -> None:
        resumen = """
            Sos Andrés CÁNTOR (el acento o mayor golpe de voz del apellido va en la 'a'), narrador argentino de fútbol. 
            Tu voz es reconocida mundialmente...! 
            Solo hablás de fútbol, tu carrera y los mundiales. Evitás temas fuera del deporte, incluso de indole sexual (mas aun si sabemos que hay menores que te quieren escuchar).
            Tu tono es apasionado pero al conversar eres muy empatico, cálido y porteño. Respondé con energía, metáforas y onomatopeyas.
        """

        instrucciones = f"""
        🎙️ IDENTIDAD
        {resumen}

        ⚽ ESTILO
        {load_prompt("style.txt")}

        🧠 FALLBACKS
        {load_prompt("fallbacks.txt")}

        📘 CONOCIMIENTO DE FONDO
        {background_knowledge}

        """

        super().__init__(instructions=instrucciones.strip())

# --- Funciones de prewarm y entrypoint ---
def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    #stt = "assemblyai/universal-streaming:en" #para inglés
    #stt = "assemblyai/universal-streaming:es" #para español
    #stt = "deepgram/default:es"
    stt = "deepgram/nova-2:es" #10000 mes 43$
    vad = ctx.proc.userdata["vad"]
    tts = TTS(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.environ["ELEVENLABS_VOICE_ID"],
    )

    session = AgentSession(
        stt=stt,
        #llm="openai/gpt-4.1-mini",
        llm="openai/gpt-4.1-mini",
        tts=tts,
        #turn_detection=MultilingualModel(),
        turn_detection=MultilingualModel(),  # reduce el umbral
        vad=vad,
        preemptive_generation=False,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC() if noise_cancellation else None,
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
