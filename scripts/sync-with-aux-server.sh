cd ~/dev/research/sralt

rsync -aH --partial --info=progress2 --delete \
  aux-server:/home/wll/dev/research/sralt/.cache/ .cache/

rsync -aH --partial --info=progress2 --delete \
  aux-server:/home/wll/dev/research/sralt/runs/ runs/

rsync -aH --partial --info=progress2 --delete \
  docs/ aux-server:/home/wll/dev/research/sralt/docs/

rsync -aH --partial --info=progress2 --delete \
  sralt/ aux-server:/home/wll/dev/research/sralt/sralt/

rsync -aH --partial --info=progress2 --delete \
  scripts/ aux-server:/home/wll/dev/research/sralt/scripts/

rsync -aH --partial --info=progress2 --delete \
  tests/ aux-server:/home/wll/dev/research/sralt/tests/

rsync -aH --partial --info=progress2 --delete \
  todo.md aux-server:/home/wll/dev/research/sralt/todo.md
