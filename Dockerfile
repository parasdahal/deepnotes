FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y ruby-full build-essential zlib1g-dev
RUN apt-get install -y python-pygments
RUN gem install pygments.rb jekyll jekyll-paginate jekyll-seo-tag jekyll-gist kramdown rouge
# RUN gem install bundler -v "~>1.0" && gem install bundler jekyll
EXPOSE 4000
WORKDIR /site
CMD [ "jekyll", "serve", "--watch", "--force_polling", "-H", "0.0.0.0", "-P", "4000" ]