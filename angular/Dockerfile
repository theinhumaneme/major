### STAGE 1: Build ###
FROM node:18 AS build
WORKDIR /usr/src/app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build
### STAGE 2: Run ###
FROM nginx:1.17.1-alpine
COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /usr/src/app/dist/angular /usr/share/nginx/html