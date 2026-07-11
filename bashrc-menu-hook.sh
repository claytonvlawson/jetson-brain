
# >>> jetson capability menu >>>
# Opens the capability menu on interactive SSH/mosh logins only.
# The `case $- in *i*` guard keeps it out of scripted ssh commands, scp, and rsync.
export PATH="$HOME/.local/bin:$PATH"
alias menu='/mnt/nvme/brain/jetson-menu.sh'
case $- in
  *i*)
    if [ -z "${JETSON_MENU_SHOWN:-}" ] && [ -z "${TMUX:-}" ] && [ -t 1 ]; then
      export JETSON_MENU_SHOWN=1
      /mnt/nvme/brain/jetson-menu.sh
    fi
    ;;
esac
# <<< jetson capability menu <<<
