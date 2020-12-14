.PHONY: up
up:
	docker-compose up -d

.PHONY: down
down: 
	docker-compose down

.PHONY: exec
exec:
	docker exec -it comparative-abusive-lang bash
