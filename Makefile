all:
	if [ ! -d font ]; then \
		echo "Downloading fonts.."; \
		wget https://font.download/dl/font/eb-garamond-2.zip; \
		wget https://github.com/samuelngs/apple-emoji-linux/releases/latest/download/AppleColorEmoji.ttf; \
		mkdir font; \
		unzip ./eb-garamond-2.zip -d font; \
		mv ./AppleColorEmoji.ttf ./font/; \
		rm ./eb-garamond-2.zip; \
	fi; \
	echo "Copying into Docker and compiling.."; \
	docker run \
		--rm \
		-v "$(PWD)":/workspace \
		--env TYPST_FONT_PATHS=/workspace/font \
		-w /workspace 123marvin123/typst \
		typst compile präsentation.typ; \
	echo "Done. Opening with xdg-open"; \
	xdg-open ./präsentation.pdf; \

