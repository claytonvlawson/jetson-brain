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
  echo "    ${gr}1${r})  Chat with your ThesisVault   ${dm}(brain: any model, RAG + auto web)${r}"
  echo "    ${gr}2${r})  Run an Ollama model directly ${dm}(plain chat, no vault)${r}"
  echo "    ${gr}3${r})  NCDD NBA draft lab"
  echo "    ${gr}4${r})  Internet status / connectivity check"
  echo "    ${gr}5${r})  System status ${dm}(GPU, memory, power, disk)${r}"
  echo "    ${gr}6${r})  Super Saiyan Mode (SSJ mode)  ${dm}[sudo]${r}"
  echo "    ${gr}7${r})  Rebuild the vault index"
  echo "    ${gr}8${r})  Open the README"
  echo "    ${gr}9${r})  ASL sign detection   ${dm}(streams your Mac webcam -> MediaPipe hands)${r}"
  echo "    ${gr}q${r})  Quit to shell   ${dm}(type 'menu' to reopen)${r}"
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

asl_detect(){
  echo
  echo "${b}ASL sign detection${r}"
  echo "  ${dm}This Jetson runs MediaPipe Hands and classifies signs.${r}"
  echo "  ${dm}On your Mac run:  cd ~/jetson-brain && ./asl-mac.sh${r}"
  echo "  ${dm}That streams your Mac webcam frames here for landmarking.${r}"
  echo
  exec python3 "$BRAIN_ROOT/asl_server.py"
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
      4) "$BRAIN_ROOT/jetson-online.sh" test 2>/dev/null || echo "connectivity script not found"; pause ;;
      5) system_status; pause ;;
      6) echo "Going Super Saiyan. sudo may prompt for your password."
         sudo nvpmodel -m 2 && sudo jetson_clocks && echo "Now:" && nvpmodel -q 2>/dev/null | grep -i 'power mode'
         pause ;;
      7) python3 "$BRAIN_ROOT/build_index.py"; pause ;;
      8) if [ -f "$BRAIN_ROOT/README.md" ]; then less "$BRAIN_ROOT/README.md"; elif [ -f "$HOME/README.md" ]; then less "$HOME/README.md"; else echo "README not found."; sleep 1; fi ;;
      9) asl_detect; pause ;;
      q|Q|"") clear 2>/dev/null; echo "At the shell. Type ${b}menu${r} to reopen the menu, or ${b}brain${r} to chat."; break ;;
      *) echo "Unknown choice: $choice"; sleep 1 ;;
    esac
  done
}
main
