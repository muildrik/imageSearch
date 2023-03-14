$(document).ready(() => {

    $('.del_image').on('click', e => {
        let data = {
            'project' : $(e.currentTarget).data('project'),
            'image' : $(e.currentTarget).data('name')
        }
        $.post('del_image', data)
        .then(result => {
            console.log(result)
        })
        .catch(err => console.log(err))
    })

})