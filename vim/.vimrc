set number
set background=dark
set hidden
colorscheme gruvbox

"The most basic stuff
set encoding=utf-8
set nocompatible
filetype plugin indent on
syntax on
set splitbelow splitright
set timeoutlen=50
set undodir=~/.vim/undodir
set undofile
set clipboard=unnamedplus

"Fix vim cache in regards to other users on ros (fixed?)
set dir=~/.cache
set backupdir=~/.cache

"Programming stuff
nnoremap <buffer> <F6> :!python3 %<cr>
set colorcolumn=81
set textwidth=80
set autoindent
set smartindent
set tabstop=4
set shiftwidth=4
set expandtab


let data_dir = has('nvim') ? stdpath('data') . '/site' : '~/.vim'
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif
"Plugins
call plug#begin()
Plug 'nvie/vim-flake8'
Plug 'tpope/vim-commentary'
    nnoremap <C-c> :Commentary<CR>
    vnoremap <C-c> :Commentary<CR>
call plug#end()

"Use 24-bit (true-color) mode in Vim/Neovim when outside tmux.
"If you're using tmux version 2.2 or later, you can remove the outermost $TMUX check and use tmux's 24-bit color support
"(see < http://sunaku.github.io/tmux-24bit-color.html#usage > for more information.)
if (empty($TMUX))
  if (has("nvim"))
    "For Neovim 0.1.3 and 0.1.4 < https://github.com/neovim/neovim/pull/2198 >
    let $NVIM_TUI_ENABLE_TRUE_COLOR=1
  endif
  "For Neovim > 0.1.5 and Vim > patch 7.4.1799 < https://github.com/vim/vim/commit/61be73bb0f965a895bfb064ea3e55476ac175162 >
  "Based on Vim patch 7.4.1770 (`guicolors` option) < https://github.com/vim/vim/commit/8a633e3427b47286869aa4b96f2bfc1fe65b25cd >
  " < https://github.com/neovim/neovim/wiki/Following-HEAD#20160511 >
  if (has("termguicolors"))
    set termguicolors
  endif
endif
