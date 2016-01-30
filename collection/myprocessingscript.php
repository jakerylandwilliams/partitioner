<?php
$text = $_POST['text'];

$text = preg_replace('/\n/', '\\n', $text);
$text = preg_replace('/\t/', '\\t', $text);
$text = preg_replace('/\015/', '', $text);

$fp = fopen("data.txt", "a");
$savestring = $text . "\n";
fwrite($fp, $savestring);
fclose($fp);
echo "<h1>Thanks for the help!</h1>";
?>
