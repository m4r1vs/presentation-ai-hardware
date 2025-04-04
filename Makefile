all:
	if [ ! -d font ]; then \
		echo "Downloading fonts.."; \
		wget https://github.com/georgd/EB-Garamond/releases/download/nightly/EBGaramond.zip; \
		wget https://github.com/samuelngs/apple-emoji-linux/releases/latest/download/AppleColorEmoji.ttf; \
		mkdir font; \
		unzip ./EBGaramond.zip -d font; \
		mv ./AppleColorEmoji.ttf ./font/; \
		rm ./EBGaramond.zip; \
	fi; \
	echo "Copying slides into Docker and compiling.."; \
	docker run \
		--rm \
		-v "$(PWD)":/workspace \
		--env TYPST_FONT_PATHS=/workspace/font \
		-w /workspace 123marvin123/typst \
		typst compile präsentation.typ; \
	echo "Copying main paper into Docker and compiling.."; \
	docker run \
		--rm \
		-v "$(PWD)":/workspace \
		--env TYPST_FONT_PATHS=/workspace/font \
		-w /workspace 123marvin123/typst \
		typst compile hardware-beschleunigung-fuer-ml-und-ai.typ; \
	echo "Done. Opening with xdg-open"; \
	xdg-open ./präsentation.pdf; \
