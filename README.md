# Reading Estiomator with DeBERTa v2

## TODO
* 動詞の活用形をいい感じに扱いたい
* 複合語処理に対応したい

## 環境構築

uvで環境構築できます。
```bash
uv sync
```

加えて、macOSで以下のようにJUMAN++の環境構築を実施した。

```bash
brew install boost
export BOOST_ROOT=$(brew --prefix boost)

wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/jumanpp/jumanpp-1.02.tar.xz
tar xJvf jumanpp-1.02.tar.xz
cd jumanpp-1.02

./configure --with-boost=$BOOST_ROOT
make CXXFLAGS="-I/opt/homebrew/Cellar/boost/1.86.0/include"
sudo make install
```

## formatter/linter

以下を実行することでformatter/linterを適用できます。
```bash
uv run ruff format
uv run ruff check --fix
```


## License

references.jsonのデータは日本語版Wikipediaのライセンスを継承し、CC-BY-SA 4.0で公開します。
