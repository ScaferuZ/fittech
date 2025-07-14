#!/bin/bash

echo "🚀 Deploying XGFitness with Reverse Proxy"
echo "=========================================="

# Step 1: Create web-gateway network if it doesn't exist
if ! docker network ls | grep -q "web-gateway"; then
    echo "🌐 Creating web-gateway network..."
    docker network create web-gateway
else
    echo "✅ web-gateway network already exists"
fi

# Step 2: Update HTTP redirect block FIRST
echo "📝 Step 1: Update HTTP redirect block in nginx config..."
echo "   Add 'xgfitness.site www.xgfitness.site' to the server_name list in:"
echo "   ~/reverse-proxy/nginx/conf.d/default.conf"
echo ""
read -p "Press Enter when HTTP redirect is updated..."

echo "🔄 Reloading nginx..."
docker compose -f ~/reverse-proxy/docker-compose.yml exec nginx nginx -s reload

# Step 3: Generate SSL certificates
echo "🔐 Generating SSL certificates..."
docker compose -f ~/reverse-proxy/docker-compose.yml run --rm certbot certonly \
    --webroot --webroot-path /var/www/certbot \
    -d xgfitness.site -d www.xgfitness.site \
    --email endra.rocks@gmail.com --agree-tos --no-eff-email

# Step 4: Add HTTPS server block
echo "📝 Step 2: Now add the HTTPS server block from nginx-xgfitness-config.conf"
echo ""
read -p "Press Enter when HTTPS server block is added..."

# Step 5: Reload nginx with HTTPS config
echo "🔄 Reloading nginx with HTTPS config..."
docker compose -f ~/reverse-proxy/docker-compose.yml exec nginx nginx -s reload

# Step 5: Deploy XGFitness app
echo "🚀 Deploying XGFitness app..."
docker-compose down --remove-orphans || true
docker-compose up -d --build

# Step 6: Wait and verify
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ XGFitness deployed successfully!"
    echo "🌐 Access your app at: https://xgfitness.site"
    echo "🔍 Health check: https://xgfitness.site/health"
else
    echo "❌ Deployment failed!"
    docker-compose logs
    exit 1
fi

echo "🎉 Deployment complete!" 