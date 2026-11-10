$ENV{'TEXINPUTS'} = './styles//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = './styles//:' . ($ENV{'BSTINPUTS'} // '');
$out_dir = 'build';
