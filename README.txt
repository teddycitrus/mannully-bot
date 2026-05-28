===========================================================================
 HISTORICAL RAG MEMORY DISCORD BOT
===========================================================================

A Discord bot that scrapes your server's entire message history into a
SQLite/libSQL archive and uses it as a permanent Retrieval-Augmented-
Generation (RAG) memory. Ask it about anything that happened in the server
- days or years ago - and it answers from its own archive using Google
Gemini 2.5 Flash.

Retrieval is HYBRID: BM25 keyword search (great at names, in-jokes, exact
recall) is fused with Gemini embedding vector search (great at paraphrase
and concept queries) via Reciprocal Rank Fusion. Falls back gracefully to
plain BM25 while embeddings are still being backfilled.

Two ways to run it:
  * LOCAL (default)      - everything in one local file, zero hosting cost.
  * 24/7 / EXTERNAL DB   - archive lives in a hosted Turso (libSQL) database
                           and the bot runs on an always-on host, so it
                           keeps working even when your PC is off. See
                           section 11.


---------------------------------------------------------------------------
 1. WHAT YOU NEED
---------------------------------------------------------------------------
 - Python 3.10 or higher  (check: python --version)
 - A Discord account with permission to add a bot to your server
 - A Google AI Studio account for a free Gemini API key


---------------------------------------------------------------------------
 2. CREATE THE DISCORD BOT + ENABLE INTENTS  (one-time)
---------------------------------------------------------------------------
 1. Go to https://discord.com/developers/applications
 2. "New Application" -> give it a name -> Create.
 3. Left sidebar -> "Bot" -> "Add Bot" -> confirm.
 4. Under "Bot", click "Reset Token" and COPY the token. This is your
    DISCORD_BOT_TOKEN (you only see it once - keep it secret).
 5. STILL on the "Bot" page, scroll to "Privileged Gateway Intents" and
    turn ON:
        [x] MESSAGE CONTENT INTENT      (required - read message text)
        [x] SERVER MEMBERS INTENT       (recommended)
    Save changes.
 6. Left sidebar -> "OAuth2" -> "URL Generator":
        SCOPES:        [x] bot
        BOT PERMISSIONS:
            [x] View Channels
            [x] Read Message History
            [x] Send Messages
            [x] Read Messages/View Channel
 7. Copy the generated URL at the bottom, open it in a browser, pick your
    server, and authorise. The bot now appears (offline) in your server.


---------------------------------------------------------------------------
 3. GET A GEMINI API KEY  (one-time)
---------------------------------------------------------------------------
 1. Go to https://aistudio.google.com/app/apikey
 2. "Create API key" -> copy it. This is your GEMINI_API_KEY.


---------------------------------------------------------------------------
 4. CONFIGURE THE PROJECT
---------------------------------------------------------------------------
 In this project folder, create a file named exactly:  .env
 with this content (replace the placeholder values):

     DISCORD_BOT_TOKEN=paste-your-discord-bot-token-here
     GEMINI_API_KEY=paste-your-gemini-api-key-here

 Optional - run with an EXTERNAL database (needed for 24/7 hosting, see
 section 11). Leave blank for the original fully-local behaviour:

     TURSO_DATABASE_URL=libsql://your-db-name-org.turso.io
     TURSO_AUTH_TOKEN=your-turso-auth-token

 Optional tuning variables (all have sensible defaults - omit if unsure):

     DATABASE_FILE=history.db          # local file / replica cache
     GEMINI_MODEL=gemini-2.5-flash     # model to use
     COMMAND_PREFIX=!                  # prefix for !sync / !stats
     SCRAPE_CHUNK_SIZE=100             # messages fetched per page
     AUTO_SYNC_IF_EMPTY=false          # true = auto-scrape on first run
     RAG_MAX_CONTEXT_MESSAGES=40       # snippets sent to Gemini per question

 IMPORTANT: never commit or share the .env file - it contains secrets.


---------------------------------------------------------------------------
 5. INSTALL DEPENDENCIES
---------------------------------------------------------------------------
 (Optional but recommended - use a virtual environment)

     python -m venv .venv
     # Windows:
     .venv\Scripts\activate
     # macOS / Linux:
     source .venv/bin/activate

 Then install:

     pip install -r requirements.txt

 Sanity-check your credentials are readable:

     python config.py
   (prints a config summary and "Configuration OK." if the .env is valid)


---------------------------------------------------------------------------
 6. RUN THE BOT
---------------------------------------------------------------------------
     python main.py

 You should see "Logged in as <bot name>" and the bot goes online in
 Discord.


---------------------------------------------------------------------------
 7. BUILD THE HISTORICAL ARCHIVE (initial sync)
---------------------------------------------------------------------------
 In any channel the bot can see, a SERVER ADMIN types:

     !sync

 The bot walks every readable text channel and downloads the full history
 in pages of 100, saving each message (author, timestamp, content,
 channel, attachments) into the local history.db file. It posts periodic
 progress and a final summary. Large/old servers can take a while - this
 is normal. The sync is resumable: rate limits or a restart won't lose
 progress, just run !sync again to continue.

 (If you set AUTO_SYNC_IF_EMPTY=true, the bot starts this automatically
  the first time it sees an empty database - no command needed.)


---------------------------------------------------------------------------
 8. USING THE BOT
---------------------------------------------------------------------------
 - New messages are saved automatically as people chat (no action needed).

 - Ask a historical question by @-mentioning the bot:

       @YourBot who proposed a mall meetup 2 days ago?
       @YourBot what did we decide about the server rules last month?
       @YourBot when did Alex first join the voice channel chats?

   It searches the local archive (hybrid BM25 + Gemini-embedding recall,
   plus any time reference like "2 days ago" / "yesterday" / "last week"),
   then asks Gemini to answer strictly from those archived messages,
   naming the people involved.

 - Or use the command form (no mention needed):

       !ask who organised the New Year event?

 - Attach an image while mentioning the bot to have Gemini analyse it
   (multimodal): @YourBot what's in this screenshot?

 - !stats  -> shows how many messages/channels/authors are archived and
              the date range covered.


---------------------------------------------------------------------------
 9. FILES IN THIS PROJECT
---------------------------------------------------------------------------
   main.py            Bot login, gateway events, commands, RAG loop,
                      embedding backfill + live-ingest embedding
   config.py          Loads + validates .env credentials and settings
   scraper.py         Async history pull with rate-limit back-off + resume
   database.py        SQLite schema, inserts, FTS keyword/time search,
                      embedding storage
   embeddings.py      Gemini text-embedding client (gemini-embedding-001,
                      truncated to 768-dim) with batch + retry
   retrieval.py       Per-guild vector cache, cosine top-K, BM25+vector
                      Reciprocal Rank Fusion
   gemini_client.py   Gemini 2.5 Flash prompt + multimodal client
   requirements.txt   Python dependencies
   Dockerfile         Container image for always-on (24/7) hosting
   .dockerignore      Files excluded from the image (secrets, local db)
   .env.example       Template to copy to .env
   README.txt         This file
   history.db         Created on first run - local archive, OR (when Turso
                      is configured) a disposable local replica cache


---------------------------------------------------------------------------
 11. RUN IT 24/7 (EXTERNAL DATABASE + HOSTING)
---------------------------------------------------------------------------
 By default the bot only runs while this machine is on and the archive
 lives only on this disk. To make it run regardless of your computer's
 status you need TWO things: an external database AND an always-on host.

 STEP A - External database (Turso / libSQL)
 -------------------------------------------
 The archive moves to a hosted libSQL database. The bot opens it as an
 "embedded replica": a local cache file (history.db) is kept in sync with
 the remote primary, so reads stay fast and every write is durably pushed
 to Turso. The local cache is disposable - it rebuilds from Turso on boot,
 so the data survives even if the host is wiped or replaced.

   1. Install the Turso CLI:  https://docs.turso.tech/cli/installation
   2. turso auth signup
   3. turso db create mannully-bot
   4. turso db show mannully-bot --url        -> TURSO_DATABASE_URL
   5. turso db tokens create mannully-bot     -> TURSO_AUTH_TOKEN
   6. Put both values in your .env (see section 4).

 Run the bot once locally with those set and do !sync - the history now
 lives in Turso. (Already have a local history.db? Import it first with:
 `turso db shell mannully-bot < dump.sql` after `sqlite3 history.db .dump
 > dump.sql`, or just re-run !sync against the empty Turso db.)

 The free Turso tier is generous (multiple databases, billions of row
 reads/month) and is plenty for a community chat archive.

 STEP B - Always-on host
 -----------------------
 Run the bot as a long-lived "worker" (no web port needed). A Dockerfile
 is included, so any container host works. Examples:

   * Fly.io:   fly launch --no-deploy
               fly secrets set DISCORD_BOT_TOKEN=... GEMINI_API_KEY=... \
                               TURSO_DATABASE_URL=... TURSO_AUTH_TOKEN=...
               fly deploy
   * Railway / Render: new project from this repo, service type
               "Worker"/"Background Worker", add the same env vars.
   * Any VPS / Raspberry Pi:
               docker build -t mannully-bot .
               docker run -d --restart=always --env-file .env mannully-bot

 Because all state is in Turso, the host filesystem is disposable: deploys,
 restarts, and host swaps lose nothing. Run it on the host and your local
 machine can be off entirely - the bot stays online and keeps archiving.

 NOTE: a host can only run ONE instance of the bot at a time (Discord
 allows one gateway session per token). Don't also leave it running on
 your PC against the same token.


---------------------------------------------------------------------------
 12. TROUBLESHOOTING
---------------------------------------------------------------------------
 "Missing required environment variable"
     -> .env is missing or a value is blank. See section 4.

 "PrivilegedIntentsRequired" on startup
     -> You didn't enable MESSAGE CONTENT INTENT. See section 2, step 5.

 Bot replies but says it can't find anything
     -> Run !sync first (section 7). Check !stats shows messages > 0.

 !sync says "You need the Administrator permission"
     -> Only server admins can trigger a sync (protects your API usage).

 Sync is slow / pauses
     -> Normal for 3 years of logs. It backs off on rate limits and
        resumes automatically. Just leave it running, or re-run !sync
        later to continue where it left off.

 "the libSQL driver is not installed"
     -> TURSO_DATABASE_URL is set but `pip install libsql` hasn't run.
        Reinstall deps: pip install -r requirements.txt

 Startup log says "Initial libSQL sync failed"
     -> Transient network/credentials issue reaching Turso. Verify
        TURSO_DATABASE_URL / TURSO_AUTH_TOKEN, and that the token isn't
        expired (recreate with `turso db tokens create <db>`). The bot
        keeps running on the local replica and retries on the next write.

 Startup log shows the wrong storage engine
     -> Check the "Storage ..." line printed on startup. "local SQLite
        file" means TURSO_DATABASE_URL was empty/unset in the environment
        the process actually sees (common on hosts: set it as a secret,
        not just in a local .env).
===========================================================================
