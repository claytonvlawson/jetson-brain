#!/usr/bin/env bash
# Jetson Orin Nano capability menu.
# Auto-opens on an interactive SSH or mosh login (hooked from ~/.bashrc).
# Reopen anytime by typing:  menu
# Safe and idempotent. No em dashes by design.

export PATH="$HOME/.local/bin:$PATH"
BRAIN_ROOT="/mnt/nvme/brain"
NCDD_DIR="$HOME/ncdd-rag"

r=$'\033[0m'; b=$'\033[1m'; cy=$'\033[36m'; gr=$'\033[32m'; ye=$'\033[33m'; dm=$'\033[2m'

pause(){ printf "\n${dm}Press Enter to return to the menu...${r}"; read -r _; }

banner(){
  printf "%s" "$cy$b"
  echo "  ============================================================"
  echo "        JETSON ORIN NANO  .  snowbeach  .  local AI box"
  echo "  ============================================================$r"
  echo "  ${dm}10 Ollama models  |  ThesisVault RAG (brain)  |  web + phone access${r}"
}

status_line(){
  local ip pmode mem web
  ip=$(tailscale ip -4 2>/dev/null || echo "n/a")
  pmode=$(nvpmodel -q 2>/dev/null | grep -i 'power mode' | awk -F: '{print $2}' | xargs)
  [ -z "$pmode" ] && pmode="?"
  mem=$(free -h | awk '/Mem/{print $3"/"$2}')
  web=$(timeout 4 curl -s -o /dev/null -w '%{http_code}' https://duckduckgo.com 2>/dev/null)
  if [ "$web" = "200" ]; then web="${gr}online${r}"; else web="${ye}offline${r}"; fi
  printf "  ${dm}tailscale${r} %s   ${dm}power${r} %s   ${dm}mem${r} %s   ${dm}internet${r} %b\n" \
    "$ip" "$pmode" "$mem" "$web"
}

show_menu(){
  echo
  echo "  ${b}What would you like to do?${r}"
  echo "    ${gr}1${r})  Language Deprivation AI Agent   ${dm}(brain: any model, RAG + auto web)${r}"
  echo "    ${gr}2${r})  Run an Ollama model directly ${dm}(plain chat, no vault)${r}"
  echo "    ${gr}3${r})  NCDD NBA draft lab"
  echo "    ${gr}4${r})  Internet status / connectivity check"
  echo "    ${gr}5${r})  System status ${dm}(GPU, memory, power, disk)${r}"
  echo "    ${gr}6${r})  Super Saiyan Mode (SSJ mode)  ${dm}[sudo]${r}"
  echo "    ${gr}7${r})  Rebuild the vault index (with embeddings)"
  echo "    ${gr}8${r})  Open the README"
  echo "    ${gr}9${r})  Sign Studio   ${dm}(live ASL sentence reader + camera controls, web)${r}  $(studio_state)"
  echo "    ${gr}v${r})  Gemma Vision   ${dm}(live camera + object identification with gemma4:e2b, web)${r}  $(vision_state)"
  echo "    ${gr}c${r})  C920 Sentences   ${dm}(record the 212 ASL sentences on the C920, web)${r}  $(prompter_state)"
  echo "    ${gr}g${r})  LangChain agent${dm}(same vault, local models + optional Groq cloud speed tier)${r}"
  echo "    ${gr}w${r})  Review fetched web pages     ${dm}($(web_queue_count) waiting for approval)${r}"
  echo "    ${gr}q${r})  Quit to shell   ${dm}(type 'menu' to reopen)${r}"
}

# Count the web pages waiting for a human decision. Fetches land in the QUEUE
# (outside the vault root) and are not indexed until someone promotes them, so
# an un-drained queue is invisible from the chat side. Surfacing the number here
# is the only place the device tells you it is waiting on you.
web_queue_count(){
  local d="$BRAIN_ROOT/queue/_WebCache"
  [ -d "$d" ] || { echo 0; return; }
  local n; n=$(ls -1 "$d"/*.md 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

web_review(){
  if [ ! -f "$BRAIN_ROOT/web_review.py" ]; then
    echo "web_review.py is not on this device yet. Re-run deploy.sh from the Mac."
    return
  fi
  local choice stem
  while :; do
    echo
    echo "  ${b}Web review queue${r}   ${dm}$(web_queue_count) waiting${r}"
    echo "    ${gr}l${r})  List the queue"
    echo "    ${gr}s${r})  Show one page in full"
    echo "    ${gr}p${r})  Promote a page into the vault ${dm}(indexes it)${r}"
    echo "    ${gr}r${r})  Reject a page ${dm}(deletes it and denies the url)${r}"
    echo "    ${gr}x${r})  Retire an ALREADY promoted note"
    echo "    ${gr}b${r})  Back"
    printf "\n  ${b}> ${r}"; read -r choice
    case "$choice" in
      l|L) python3 "$BRAIN_ROOT/web_review.py" --list; pause ;;
      s|S) printf "  stem: "; read -r stem
           [ -n "$stem" ] && python3 "$BRAIN_ROOT/web_review.py" --show "$stem"; pause ;;
      p|P) printf "  stem to promote: "; read -r stem
           [ -n "$stem" ] && python3 "$BRAIN_ROOT/web_review.py" --promote "$stem"; pause ;;
      r|R) printf "  stem to reject: "; read -r stem
           [ -n "$stem" ] && python3 "$BRAIN_ROOT/web_review.py" --reject "$stem"; pause ;;
      # Retire is the destructive one: it removes the note, its json, and every
      # cached answer citing it. web_review.py prints the restart instruction
      # afterwards, and that restart is required, not advisory, because the
      # running services keep serving the retired answer from memory.
      x|X) printf "  stem to retire: "; read -r stem
           [ -n "$stem" ] && python3 "$BRAIN_ROOT/web_review.py" --retire "$stem"; pause ;;
      b|B|"") return ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}

ollama_pick(){
  local models=() i=1 n
  while IFS= read -r line; do models+=("$line"); done < <(ollama list 2>/dev/null | awk 'NR>1{print $1}')
  if [ ${#models[@]} -eq 0 ]; then echo "No Ollama models found (is the service up?)."; return; fi
  echo; echo "Installed models:"
  for m in "${models[@]}"; do printf "   %2d) %s\n" "$i" "$m"; i=$((i+1)); done
  printf "Pick a number (Enter to cancel): "; read -r n
  if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#models[@]} ]; then
    echo "Starting ${models[$((n-1))]} . Type /bye to leave."
    ollama run "${models[$((n-1))]}"
  fi
}

system_status(){
  echo; echo "${b}== power mode ==${r}";   nvpmodel -q 2>/dev/null | head -3
  echo; echo "${b}== memory ==${r}";       free -h
  echo; echo "${b}== disk ==${r}";         df -h /mnt/nvme / 2>/dev/null | tail -2
  echo; echo "${b}== soc / gpu (2s sample) ==${r}"; timeout 2 tegrastats 2>/dev/null | tail -1
  echo; echo "${b}== tailscale ==${r}";    tailscale ip -4 2>/dev/null
}

ncdd_lab(){
  if [ -d "$NCDD_DIR" ]; then
    echo; echo "NCDD NBA draft lab lives in ${b}$NCDD_DIR${r}"
    echo "Scripts:"; ls "$NCDD_DIR"/*.py 2>/dev/null | sed 's#.*/#   #'
    echo; echo "Usage of the RAG entrypoint:"
    ( cd "$NCDD_DIR" && python3 ncdd_rag.py --help 2>&1 | head -20 )
    echo; echo "${dm}Run it yourself, e.g.:  cd $NCDD_DIR && python3 ncdd_rag.py --list-models${r}"
  else
    echo "ncdd-rag not found in your home directory."
  fi
}

# langchain_agent.py's own CLI (interactive() / main() in that file) --
# same vault and local models as option 1, plus an optional Groq cloud tier
# for short/simple questions. groq.env.local is the same git-ignored,
# chmod-600 key file make_cloud_chat_model() reads GROQ_API_KEY from; its
# absence just means the router stays local-only (see choose_tier()).
langchain_repl(){
  echo
  echo "${b}LangChain agent${r}   ${dm}(local llama3.2:3b/qwen2.5:7b + optional Groq cloud speed tier)${r}"
  if [ -f "$BRAIN_ROOT/groq.env.local" ]; then
    set -a; source "$BRAIN_ROOT/groq.env.local"; set +a
    echo "  ${dm}GROQ_API_KEY loaded from groq.env.local -- short questions may route to Groq.${r}"
  else
    echo "  ${dm}No groq.env.local found -- running local-only. See groq.env.example.${r}"
  fi
  echo
  ( cd "$BRAIN_ROOT" && python3 langchain_agent.py --interactive )
}

# Sign Studio: sign_studio.py in the ASL pipeline dir (built from the Mac's
# ~/Desktop/Dream/asl-sentence-trainer). It runs detached, so it keeps going after
# the menu or the SSH session closes. It binds loopback (for the Mac's studio.sh tunnel,
# http://localhost:8092) plus the USB-C link address 192.168.55.1 when that link is up.
# Never the tailnet or LAN: the page controls the camera and has no login.
ASL_DIR="/mnt/nvme/mediapipe-hand"
STUDIO_PORT=8092

studio_pid(){ pgrep -f '[s]ign_studio\.py' | head -1; }
studio_state(){ [ -n "$(studio_pid)" ] && echo "${gr}[running]${r}" || echo ""; }
usb_up(){ ip -4 -o addr show 2>/dev/null | grep -q ' 192\.168\.55\.1/'; }
studio_host(){ usb_up && echo 192.168.55.1 || echo 127.0.0.1; }

studio_start(){
  if [ ! -f "$ASL_DIR/sign_studio.py" ]; then
    echo "sign_studio.py is not in $ASL_DIR. Run ./studio.sh deploy from the Mac."
    return
  fi
  local host binds; host=$(studio_host)
  binds=127.0.0.1; usb_up && binds="127.0.0.1,192.168.55.1"
  "$ASL_DIR/asl_cam.sh" stop >/dev/null 2>&1      # signcam holds the camera
  pkill -f '[g]emma_vision\.py'                   # so does Gemma Vision
  pkill -f '[s]entence_decoder\.py'; pkill -f '[s]ign_studio\.py'; pkill -f '[l]ive_prompter\.py'; sleep 1
  mkdir -p "$ASL_DIR/logs"
  # Redirect the SUBSHELL, not just python: with "cd && python3 &" the & backgrounds the
  # whole list, and that extra bash kept the menu's stdout open, so SSH never closed.
  ( cd "$ASL_DIR" && setsid python3 sign_studio.py --host "$binds" --port "$STUDIO_PORT" & ) \
      </dev/null >"$ASL_DIR/logs/sign_studio.log" 2>&1
  printf "  starting (models take ~10s)"
  local i
  for i in $(seq 1 30); do
    curl -s -m 1 -o /dev/null "http://$host:$STUDIO_PORT/api/state" && break
    printf "."; sleep 1
  done
  echo
  if curl -s -m 2 -o /dev/null "http://$host:$STUDIO_PORT/api/state"; then
    echo "  ${gr}Sign Studio is up.${r}"
    if [ "$host" = "192.168.55.1" ]; then
      echo "  On the Mac (USB-C cable):  ${b}http://192.168.55.1:$STUDIO_PORT/${r}"
      echo "  ${dm}or ./studio.sh start in ~/Desktop/Dream/asl-sentence-trainer for http://localhost:$STUDIO_PORT/${r}"
    else
      echo "  Loopback only (no USB-C link). On the Mac run:"
      echo "    ssh -N -L $STUDIO_PORT:127.0.0.1:$STUDIO_PORT user@<jetson>   then open http://localhost:$STUDIO_PORT/"
    fi
    echo "  ${dm}It keeps running after you leave the menu. Stop it with 9 then x.${r}"
  else
    echo "  ${ye}Not answering. Last log lines:${r}"; tail -15 "$ASL_DIR/logs/sign_studio.log"
  fi
}

sign_studio(){
  local choice
  while :; do
    local pid host; pid=$(studio_pid); host=$(studio_host)
    echo
    if [ -n "$pid" ]; then
      echo "  ${b}Sign Studio${r}   ${gr}running${r} ${dm}(pid $pid)  http://$host:$STUDIO_PORT/${r}"
    else
      echo "  ${b}Sign Studio${r}   ${dm}stopped${r}"
    fi
    echo "    ${gr}s${r})  Start / restart   ${dm}(stops signcam first, one reader per camera)${r}"
    echo "    ${gr}x${r})  Stop"
    echo "    ${gr}l${r})  Show the log"
    echo "    ${gr}r${r})  Recent reads   ${dm}(logs/decoded.jsonl)${r}"
    echo "    ${gr}b${r})  Back"
    printf "\n  ${b}> ${r}"; read -r choice
    case "$choice" in
      s|S) studio_start; pause ;;
      x|X) pkill -f '[s]ign_studio\.py' && echo "  stopped" || echo "  was not running"; pause ;;
      l|L) tail -30 "$ASL_DIR/logs/sign_studio.log" 2>/dev/null || echo "  no log yet"; pause ;;
      r|R) if [ -f "$ASL_DIR/logs/decoded.jsonl" ]; then
             tail -15 "$ASL_DIR/logs/decoded.jsonl" | python3 -c '
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    tgt = " (target: %s, %s)" % (" ".join(d["target"]), "exact" if d.get("exact") else "%d%%" % round(100 * (d.get("score") or 0))) if d.get("target") else ""
    print("  %s  %s%s" % (d.get("wall", "")[11:19], " ".join(d.get("gloss") or []) or "(nothing)", tgt))'
           else echo "  no reads logged yet"; fi
           pause ;;
      b|B|"") return ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}

# Gemma Vision: gemma_vision.py in /mnt/nvme/gemma-vision (source in the Mac repo's
# vision/ folder, copied here by hand, not by deploy.sh). Same rules as Sign Studio:
# detached so it outlives the menu, bound to loopback plus the USB-C address only
# because the page shows the camera and has no login, and it takes the camera from
# signcam / Sign Studio (one reader per camera). gemma4:e2b loads on the first
# identification and Ollama releases it after 5 idle minutes.
VISION_DIR="/mnt/nvme/gemma-vision"
VISION_PORT=8093

vision_pid(){ pgrep -f '[g]emma_vision\.py' | head -1; }
vision_state(){ [ -n "$(vision_pid)" ] && echo "${gr}[running]${r}" || echo ""; }

vision_start(){
  if [ ! -f "$VISION_DIR/gemma_vision.py" ]; then
    echo "gemma_vision.py is not in $VISION_DIR. Copy vision/gemma_vision.py from the Mac repo."
    return
  fi
  if ! ollama list 2>/dev/null | grep -q '^gemma4:e2b'; then
    echo "  ${ye}gemma4:e2b is not pulled.${r} Run:  ollama pull gemma4:e2b-it-q4_K_M   ${dm}(4.6 GB)${r}"
    return
  fi
  local host binds; host=$(studio_host)
  binds=127.0.0.1; usb_up && binds="127.0.0.1,192.168.55.1"
  "$ASL_DIR/asl_cam.sh" stop >/dev/null 2>&1      # signcam holds the camera
  pkill -f '[s]entence_decoder\.py'; pkill -f '[s]ign_studio\.py'; pkill -f '[g]emma_vision\.py'; pkill -f '[l]ive_prompter\.py'; sleep 1
  mkdir -p "$VISION_DIR/logs"
  ( cd "$VISION_DIR" && setsid python3 gemma_vision.py --host "$binds" --port "$VISION_PORT" & ) \
      </dev/null >"$VISION_DIR/logs/gemma_vision.log" 2>&1
  printf "  starting"
  local i
  for i in $(seq 1 15); do
    curl -s -m 1 -o /dev/null "http://$host:$VISION_PORT/health" && break
    printf "."; sleep 1
  done
  echo
  if curl -s -m 2 -o /dev/null "http://$host:$VISION_PORT/health"; then
    echo "  ${gr}Gemma Vision is up.${r}"
    if [ "$host" = "192.168.55.1" ]; then
      echo "  On the Mac (USB-C cable):  ${b}http://192.168.55.1:$VISION_PORT/${r}"
    else
      echo "  Loopback only (no USB-C link). On the Mac run:"
      echo "    ssh -N -L $VISION_PORT:127.0.0.1:$VISION_PORT user@<jetson>   then open http://localhost:$VISION_PORT/"
    fi
    echo "  ${dm}The video is live. Each identification takes about 20 to 80 s on this board.${r}"
    echo "  ${dm}It keeps running after you leave the menu. Stop it with v then x.${r}"
  else
    echo "  ${ye}Not answering. Last log lines:${r}"; tail -15 "$VISION_DIR/logs/gemma_vision.log"
  fi
}

gemma_vision(){
  local choice
  while :; do
    local pid host; pid=$(vision_pid); host=$(studio_host)
    echo
    if [ -n "$pid" ]; then
      echo "  ${b}Gemma Vision${r}   ${gr}running${r} ${dm}(pid $pid)  http://$host:$VISION_PORT/${r}"
    else
      echo "  ${b}Gemma Vision${r}   ${dm}stopped${r}"
    fi
    echo "    ${gr}s${r})  Start / restart   ${dm}(stops Sign Studio first, one reader per camera)${r}"
    echo "    ${gr}x${r})  Stop"
    echo "    ${gr}l${r})  Show the log"
    echo "    ${gr}b${r})  Back"
    printf "\n  ${b}> ${r}"; read -r choice
    case "$choice" in
      s|S) vision_start; pause ;;
      x|X) pkill -f '[g]emma_vision\.py' && echo "  stopped" || echo "  was not running"; pause ;;
      l|L) tail -30 "$VISION_DIR/logs/gemma_vision.log" 2>/dev/null || echo "  no log yet"; pause ;;
      b|B|"") return ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}

# C920 Sentences: live_prompter.py in the ASL pipeline dir. Serves the 212-sentence pool,
# records only from the C920 on this board, and writes every clip to the microSD under
# $PROMPTER_ROOT/<session>/. It binds loopback ONLY (the page controls the camera and
# has no login), so the Mac reaches it through an SSH tunnel: ./studio.sh prompter start
# on the Mac does the tunnel and opens the page. Detached, so it outlives the menu.
PROMPTER_PORT=8095
PROMPTER_SESSION=pool212
PROMPTER_ROOT="/home/snowbeach/asl-live"

prompter_pid(){ pgrep -f '[l]ive_prompter\.py' | head -1; }
prompter_state(){ [ -n "$(prompter_pid)" ] && echo "${gr}[running]${r}" || echo ""; }

prompter_start(){
  if [ ! -f "$ASL_DIR/live_prompter.py" ]; then
    echo "live_prompter.py is not in $ASL_DIR. Run ./studio.sh deploy from the Mac."
    return
  fi
  local cam; cam=$(readlink -f /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920*-video-index0 2>/dev/null | head -1)
  if [ -z "$cam" ] || [ ! -e "$cam" ]; then
    echo "  ${ye}No C920 found.${r} Plug it into the Jetson and try again."
    return
  fi
  # never cut off a Sign Studio Record take
  local phase
  phase=$(curl -s -m 2 "http://127.0.0.1:$STUDIO_PORT/api/state" 2>/dev/null | python3 -c '
import json, sys
try:
    print((json.load(sys.stdin).get("rec") or {}).get("phase") or "")
except Exception:
    print("")')
  case "$phase" in
    countdown|recording|review)
      echo "  ${ye}Sign Studio is mid-take (Record: $phase).${r} Finish or redo it, then start this."
      return ;;
  esac
  "$ASL_DIR/asl_cam.sh" stop >/dev/null 2>&1      # one reader per camera
  pkill -f '[s]entence_decoder\.py'; pkill -f '[s]ign_studio\.py'
  pkill -f '[g]emma_vision\.py'; pkill -f '[l]ive_prompter\.py'; sleep 1
  mkdir -p "$ASL_DIR/logs"
  ( cd "$ASL_DIR" && setsid python3 live_prompter.py --session "$PROMPTER_SESSION" & ) \
      </dev/null >"$ASL_DIR/logs/live_prompter.log" 2>&1
  printf "  starting"
  local i
  for i in $(seq 1 20); do
    curl -sf -m 1 -o /dev/null "http://127.0.0.1:$PROMPTER_PORT/api/state" && break
    printf "."; sleep 1
  done
  echo
  if curl -sf -m 2 -o /dev/null "http://127.0.0.1:$PROMPTER_PORT/api/state"; then
    echo "  ${gr}C920 Sentences is up.${r} Session ${b}$PROMPTER_SESSION${r}, clips in $PROMPTER_ROOT/$PROMPTER_SESSION/"
    echo "  Loopback only. On the Mac, easiest:"
    echo "    ${b}cd ~/Desktop/Dream/asl-sentence-trainer && ./studio.sh prompter start${r}   ${dm}(tunnel + opens the page)${r}"
    echo "  or by hand:  ssh -N -L $PROMPTER_PORT:127.0.0.1:$PROMPTER_PORT user@<jetson>   then http://localhost:$PROMPTER_PORT/"
    echo "  ${dm}It keeps running after you leave the menu. Stop it with c then x.${r}"
  else
    echo "  ${ye}Not answering. Last log lines:${r}"; tail -15 "$ASL_DIR/logs/live_prompter.log"
  fi
}

c920_sentences(){
  local choice
  while :; do
    local pid; pid=$(prompter_pid)
    echo
    if [ -n "$pid" ]; then
      echo "  ${b}C920 Sentences${r}   ${gr}running${r} ${dm}(pid $pid)  localhost:$PROMPTER_PORT via tunnel${r}"
    else
      echo "  ${b}C920 Sentences${r}   ${dm}stopped${r}"
    fi
    echo "    ${gr}s${r})  Start / restart   ${dm}(stops Sign Studio and Gemma Vision first, one reader per camera)${r}"
    echo "    ${gr}x${r})  Stop"
    echo "    ${gr}l${r})  Show the log and last takes"
    echo "    ${gr}b${r})  Back"
    printf "\n  ${b}> ${r}"; read -r choice
    case "$choice" in
      s|S) prompter_start; pause ;;
      x|X) pkill -f '[l]ive_prompter\.py' && echo "  stopped" || echo "  was not running"; pause ;;
      l|L) tail -30 "$ASL_DIR/logs/live_prompter.log" 2>/dev/null || echo "  no log yet"
           echo "  --- last takes:"; tail -5 "$PROMPTER_ROOT/$PROMPTER_SESSION/takes.jsonl" 2>/dev/null || echo "  none yet"
           pause ;;
      b|B|"") return ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}

main(){
  while true; do
    clear 2>/dev/null
    banner
    status_line
    show_menu
    printf "\n  ${b}> ${r}"; read -r choice
    case "$choice" in
      1) brain ;;
      2) ollama_pick; pause ;;
      3) ncdd_lab; pause ;;
      4) if [ -x "$BRAIN_ROOT/jetson-online.sh" ]; then "$BRAIN_ROOT/jetson-online.sh" test; else echo "connectivity script not found"; fi; pause ;;
      5) system_status; pause ;;
      6) echo "Going Super Saiyan. sudo may prompt for your password."
         sudo nvpmodel -m 2 && sudo jetson_clocks && echo "Now:" && nvpmodel -q 2>/dev/null | grep -i 'power mode'
         pause ;;
      7) # --embed keeps the dense block. A bare build_index.py run rebuilds
         # BM25-only and drops the vectors hybrid retrieval needs, which is why
         # build_index.py now refuses that unless you pass --no-embed.
         python3 "$BRAIN_ROOT/build_index.py" --embed; pause ;;
      8) if [ -f "$BRAIN_ROOT/README.md" ]; then less "$BRAIN_ROOT/README.md"; elif [ -f "$HOME/README.md" ]; then less "$HOME/README.md"; else echo "README not found."; sleep 1; fi ;;
      9) sign_studio ;;
      v|V) gemma_vision ;;
      c|C) c920_sentences ;;
      g|G) langchain_repl; pause ;;
      w|W) web_review ;;
      q|Q|"") clear 2>/dev/null; echo "At the shell. Type ${b}menu${r} to reopen the menu, or ${b}brain${r} to chat."; break ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}
main
