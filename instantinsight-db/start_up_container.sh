#!/bin/bash

echo "🚀 Starting containers (database + Redis cache)..."
docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

echo "🔍 Checking Redis cache connection..."
redis_ready=$(docker exec redis-stack redis-cli ping 2>/dev/null || echo "FAILED")

if [ "$redis_ready" = "PONG" ]; then
    echo "✅ Redis cache ready"
else
    echo "❌ Redis cache not responding, retrying in 5s..."
    sleep 5
    redis_ready=$(docker exec redis-stack redis-cli ping 2>/dev/null || echo "FAILED")
    if [ "$redis_ready" = "PONG" ]; then
        echo "✅ Redis cache ready (retry successful)"
    else
        echo "❌ Redis cache failed to start - check docker logs redis-stack"
        exit 1
    fi
fi

echo "🔍 Checking PostgreSQL database..."
db_ready=$(docker exec db pg_isready -U postgres 2>/dev/null || echo "FAILED")

if [[ "$db_ready" == *"accepting connections"* ]]; then
    echo "✅ PostgreSQL database ready"
else
    echo "❌ PostgreSQL database not ready - check docker logs db"
    exit 1
fi

echo "📊 Checking Redis cache index..."
index_exists=$(docker exec redis-stack redis-cli FT.INFO idx:semantic_cache 2>/dev/null | grep -c "index_name" || echo "0")

if [ "$index_exists" -gt 0 ]; then
    echo "✅ Redis vector index already exists"
else
    echo "📝 Vector index will be created automatically on first cache access"
fi

echo "🎉 All services ready!"
echo "📊 Redis cache available at: localhost:6379"
echo "🖥️ RedisInsight UI: http://localhost:5540"
echo "🐘 PostgreSQL: localhost:5432"
echo "🔧 pgAdmin: http://localhost:5050"